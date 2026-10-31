"""Symptom word -> department, by a plain algorithm (owner, 2026-10-31: "ছুলি" -- the skin disease -- found no department; "think of a pure algorithm where symptoms will be told and you need to point to the
correct department and doctor, for every symptom").

How it works (no model, no guess):
  * ONE table: department -> weighted words (Bengali, Hindi, English and the Bengali/Hindi written in Latin letters, plus the ways a phone recogniser spells them).
      weight 2  a disease or a specialist's own word  (ছুলি, চুলকানি, migraine, hernia, ইউটিআই)  -- decides on its own
      weight 1  a symptom or a body part              (কান, দাঁত, headache, chest)               -- decides when nothing stronger is said
      weight 0.5 a vague word                         (ব্যথা, pain, শরীর খারাপ)                    -- only when nothing else is said
  * The sentence is cut into words. Every word is "folded" (long/short vowels, the three sibilants, ণ/ন, the nukta, doubled Latin letters, zero-width marks are made equal) so a recogniser's
    spelling of the same word still meets the table.
  * A word of the sentence meets a table word when it is the same, the same with an ordinary ending (ছুলিতে, ছুলির, daanter), or -- for words of five letters or more -- one letter away.
    A table entry of several words matches when its words come in order with at most two words between.
  * Each department's score is its strongest weight (then how many words hit). The strongest department wins; two departments with the same score are returned together (the caller is asked
    which one), exactly as the keyword table does.

This is the FALLBACK of booking_service.route_department: the clinic's own keyword table (editable by the clinic) and the lay-doctor phrases are asked first; only a sentence they could not place comes here.
It decides a DEPARTMENT only -- the doctors come from the database. Pure text in, department names out.
"""

from __future__ import annotations

import re
import unicodedata

# ------------------------------------------------------------------------------------------------------------------------------ the table
# department -> {weight: [words / phrases]}
LEXICON: dict[str, dict[float, list[str]]] = {
    "Dermatology": {
        2: [
            # Bengali
            "ছুলি", "ছুলির", "চুলকানি", "চুলকায়", "চুলকাচ্ছে", "চুলকাচ্ছে", "চুলকাই", "খোস", "খোসপাঁচড়া", "পাঁচড়া", "দাদ", "ফুসকুড়ি", "ফুসকুরি", "ফুসকি", "ঘামাচি", "ব্রণ", "পিম্পল",
            "মেছতা", "মেচেতা", "শ্বেতী", "ধবল", "সাদা দাগ", "কালো দাগ", "কালচে দাগ", "লাল দাগ", "গায়ে দাগ", "মুখে দাগ", "চামড়া", "চামড়ার", "ত্বক", "ত্বকের", "স্কিন", "একজিমা", "সোরিয়াসিস",
            "আমবাত", "ছত্রাক", "ফাঙ্গাস", "ফাংগাস", "খুশকি", "চুল পড়া", "চুল পড়ছে", "চুল উঠছে", "টাক", "আঁচিল", "আচিল", "ফোসকা", "র‍্যাশ", "র‍্যাশ", "গায়ে গুটি", "গুটি গুটি",
            # Hindi
            "दाद", "खाज", "खुजली", "खाज खुजली", "फुंसी", "फुंसियां", "फुंसियाँ", "मुंहासे", "मुहांसे", "कील मुंहासे", "चकत्ते", "एक्जिमा", "एक्ज़िमा", "सोरायसिस", "सफेद दाग", "सफ़ेद दाग",
            "झाइयां", "झाइयाँ", "रूसी", "बाल झड़", "बाल झड़ना", "बाल गिर", "गंजापन", "त्वचा", "चमड़ी", "स्किन", "पित्ती", "फंगस", "फफोले",
            # English
            "skin", "itch", "itching", "itchy", "rash", "rashes", "acne", "pimple", "pimples", "eczema", "psoriasis", "ringworm", "fungal", "fungus", "dandruff", "hair fall", "hair loss", "baldness",
            "hives", "wart", "warts", "vitiligo", "white patch", "white patches", "melasma", "scabies", "pigmentation", "dark spots", "blisters", "dermatologist", "dermatology",
            # Bengali / Hindi in Latin letters
            "chuli", "chulkani", "chulkay", "chulkachhe", "chulkachche", "khos", "panchra", "fuskuri", "ghamachi", "bron", "mechta", "shweti", "sweti", "sada dag", "kalo dag", "chamra", "twak",
            "chul pora", "chul porche", "khushki", "achil", "khujli", "khaj", "chakatte", "munhase",
        ],
        1: ["দাগ", "দাগ হয়ে", "दाग", "dag", "gaye dag", "spot", "spots"],
    },
    "Cardiology": {
        2: [
            "হার্ট", "হার্টের", "হৃদরোগ", "হৃদপিণ্ড", "হৃদযন্ত্র", "হার্ট অ্যাটাক", "ধড়ফড়", "ধরফর", "বুক ধড়ফড়", "বুকে ব্যথা", "বুক ব্যথা", "বুকে চাপ", "বুক চাপ", "বুকে চিনচিন", "প্রেসার",
            "প্রেশার", "ব্লাড প্রেসার", "ব্লাড প্রেশার", "বিপি", "হাই প্রেসার", "লো প্রেসার", "উচ্চ রক্তচাপ", "রক্তচাপ", "কোলেস্টেরল", "ইসিজি", "ইকো", "ব্লকেজ", "বাইপাস", "অ্যাঞ্জিও", "অ্যানজিও",
            "दिल", "दिल की", "धड़कन", "सीने में दर्द", "सीने में दबाव", "सीने में", "हार्ट अटैक", "ब्लड प्रेशर", "बीपी", "हाई बीपी", "लो बीपी", "कोलेस्ट्रॉल", "ब्लॉकेज", "बाईपास",
            "heart", "chest pain", "palpitation", "palpitations", "blood pressure", "bp", "hypertension", "cholesterol", "ecg", "echo", "blockage", "bypass", "angioplasty", "cardiologist", "cardiology",
            "buke byatha", "buk dhorfor", "buke chap", "presar", "pressure",
        ],
        1: ["বুক", "বুকে", "सीना", "chest"],
    },
    "Gynaecology & Obstetrics": {
        2: [
            "মাসিক", "মাসিকের", "পিরিয়ড", "পিরিয়ডস", "গর্ভবতী", "প্রেগন্যান্ট", "প্রেগন্যান্সি", "গর্ভাবস্থা", "সাদা স্রাব", "সাদাস্রাব", "জরায়ু", "ডিম্বাশয়", "পিসিওডি", "পিসিওএস", "ডেলিভারি",
            "সিজার", "বন্ধ্যাত্ব", "বাচ্চা হচ্ছে না", "মা হতে পারছি না", "ইউটেরাস", "ওভারি", "ফাইব্রয়েড", "রজঃস্রাব", "ঋতুস্রাব", "মেনোপজ", "মিসক্যারেজ", "গাইনি", "গাইনোকোলজিস্ট", "স্তনে চাকা",
            "माहवारी", "मासिक धर्म", "पीरियड", "पीरियड्स", "गर्भवती", "प्रेग्नेंट", "प्रेगनेंट", "गर्भावस्था", "सफेद पानी", "बच्चेदानी", "अंडाशय", "पीसीओडी", "डिलीवरी", "बांझपन", "मेनोपॉज", "गर्भपात",
            "pregnant", "pregnancy", "period", "periods", "menstrual", "uterus", "ovary", "pcod", "pcos", "white discharge", "fibroid", "delivery", "infertility", "menopause", "miscarriage", "breast lump",
            "gynaecologist", "gynecologist", "gynae", "masik", "garbhoboti", "safed pani",
        ],
    },
    "Orthopaedics": {
        2: [
            "হাড়", "হাড়ের", "জয়েন্ট", "জয়েন্টের", "গাঁটে ব্যথা", "গাঁটের ব্যথা", "বাত", "বাতের", "আর্থ্রাইটিস", "কোমর ব্যথা", "কোমরে ব্যথা", "ঘাড় ব্যথা", "হাঁটু", "হাঁটুতে", "কাঁধ", "কাঁধে", "মচকে",
            "মচকানো", "ভেঙে", "ভেঙ্গে", "ভাঙা", "ভাঙ্গা", "ফ্র্যাকচার", "প্লাস্টার", "স্পনডাইলাইটিস", "সায়াটিকা", "গোড়ালি", "গোড়ালিতে", "ডিসলোকেশন",
            "हड्डी", "हड्डियों", "जोड़", "जोड़ों", "गठिया", "कमर दर्द", "घुटना", "घुटने", "कंधा", "कंधे", "मोच", "फ्रैक्चर", "सायटिका", "स्पॉन्डिलाइटिस", "एड़ी", "प्लास्टर",
            "bone", "bones", "joint", "joints", "arthritis", "back pain", "neck pain", "knee", "shoulder", "sprain", "fracture", "dislocation", "sciatica", "spondylitis", "heel pain", "orthopaedic",
            "orthopedic", "haar", "hader", "gnather byatha", "komor byatha", "hanu", "kadh",
        ],
    },
    "ENT": {
        2: [
            "টনসিল", "সাইনাস", "নাক বন্ধ", "নাক দিয়ে জল", "নাক দিয়ে রক্ত", "নাকে রক্ত", "কানে কম শুনি", "কানে শুনতে", "কান পাকা", "কান পাকছে", "কানে পুঁজ", "কানের পর্দা", "ঘ্রাণ পাই না",
            "গলা ব্যথা", "গলা খুসখুস", "গলা বসে", "স্বর বসে", "গলার স্বর", "ঢোক গিলতে", "নাক ডাকা", "কানে ভোঁ ভোঁ", "কানে আওয়াজ", "টিনিটাস", "টনসিলাইটিস", "ই এন টি", "ইএনটি",
            "टॉन्सिल", "साइनस", "सुनाई", "कान बहना", "आवाज बैठ", "नकसीर", "गले में दर्द", "गला खराब", "कान में दर्द", "नाक बंद",
            "tonsil", "tonsils", "sinus", "hearing", "tinnitus", "nosebleed", "hoarse voice", "snoring", "ent", "otitis", "sore throat", "ear pain", "ear ache", "earache",
        ],
        1: [
            "কান", "কানে", "কানের", "নাক", "নাকে", "নাকের", "গলা", "গলায়", "গলার", "নাখ", "নাখে",
            "कान", "नाक", "गला", "गले", "ear", "ears", "nose", "throat", "kan", "nak", "gola",
        ],
    },
    "Dentistry": {
        2: [
            "দাঁত", "দাত", "দাঁতের", "দাঁতে", "মাড়ি", "মাড়ী", "মাড়িতে", "দন্ত", "ডেন্টিস্ট", "কেভিটি", "দাঁতে পোকা", "রুট ক্যানাল", "ব্রেসেস", "দাঁত তোলা", "আক্কেল দাঁত",
            "दांत", "दाँत", "दांतों", "मसूड़े", "मसूड़ा", "मसूड़ों", "डेंटिस्ट", "कैविटी", "रूट कैनाल",
            "tooth", "teeth", "gum", "gums", "dentist", "dental", "cavity", "root canal", "braces", "wisdom tooth", "daat", "dante", "danter",
        ],
    },
    "Paediatrics": {
        2: [
            "নবজাতক", "সদ্যোজাত", "শিশু", "শিশুর", "বাচ্চার ওজন", "দুধ খায় না", "ইমিউনাইজেশন", "বাচ্চার টিকা", "বাচ্চাদের টিকা", "শিশুরোগ",
            "नवजात", "शिशु", "बच्चे को टीका", "टीकाकरण", "newborn", "infant", "paediatrician", "pediatrician", "baby vaccination", "immunization", "immunisation",
        ],
    },
    "Diabetology & Endocrinology": {
        2: [
            "সুগার", "সুগারের", "ডায়াবেটিস", "ডায়াবেটিক", "মধুমেহ", "থাইরয়েড", "থাইরয়ড", "হরমোন", "ইনসুলিন", "গলগণ্ড", "গয়টার", "বারবার প্রস্রাব পায়", "খুব তেষ্টা পায়", "ওজন বেড়ে যাচ্ছে",
            "शुगर", "मधुमेह", "डायबिटीज़", "डायबिटीज", "थायराइड", "हार्मोन", "इंसुलिन", "मोटापा", "वजन बढ़",
            "sugar", "diabetes", "diabetic", "thyroid", "hormone", "hormones", "insulin", "obesity", "weight gain", "goitre", "goiter", "endocrinologist",
        ],
    },
    "General Surgery": {
        2: [
            "হার্নিয়া", "হারনিয়া", "অ্যাপেন্ডিক্স", "অ্যাপেনডিক্স", "পাইলস", "অর্শ", "ফিস্টুলা", "ফিসার", "পিত্তথলি", "গলস্টোন", "গলব্লাডার", "ফোঁড়া", "ফোড়া", "টিউমার", "চাকা", "হাইড্রোসিল",
            "ভ্যারিকোজ", "অপারেশন", "সার্জারি", "সার্জন", "গ্যাংগ্রিন", "পেটে চাকা",
            "बवासीर", "हर्निया", "हार्निया", "अपेंडिक्स", "फिस्टुला", "फोड़ा", "गांठ", "ट्यूमर", "ऑपरेशन", "सर्जरी",
            "hernia", "appendix", "appendicitis", "piles", "haemorrhoids", "hemorrhoids", "fistula", "fissure", "gallbladder", "gall bladder", "gallstone", "abscess", "boil", "lump", "tumour", "tumor",
            "hydrocele", "varicose", "surgery", "surgeon",
        ],
    },
    "Neurology": {
        2: [
            "খিঁচুনি", "খিচুনি", "মৃগী", "মৃগি", "মাইগ্রেন", "স্ট্রোক", "প্যারালাইসিস", "পক্ষাঘাত", "মাথা ব্যথা", "মাথাব্যথা", "মাথা ধরা", "মাথা ধরে", "ঝিনঝিন", "ঝিন ঝিন", "ঝিম ঝিম",
            "অসাড়", "অবশ", "হাত কাঁপে", "কাঁপুনি", "ভুলে যাই", "স্মৃতিভ্রংশ", "মুখ বেঁকে", "কথা জড়িয়ে", "স্নায়ু", "নার্ভ", "পার্কিনসন", "ব্রেন", "মস্তিষ্ক",
            "मिर्गी", "दौरे", "माइग्रेन", "लकवा", "स्ट्रोक", "सिर दर्द", "सिरदर्द", "झनझनाहट", "सुन्न", "कंपकंपी", "भूलने", "पार्किंसन", "दिमाग",
            "headache", "migraine", "seizure", "seizures", "fits", "epilepsy", "stroke", "paralysis", "numbness", "tingling", "tremor", "memory loss", "parkinson", "neurologist", "neurology",
        ],
        1: ["মাথা ঘোরা", "মাথা ঘুরছে", "চক্কর", "चक्कर", "dizziness", "dizzy", "vertigo"],
    },
    "Neurosurgery": {
        2: ["ব্রেন টিউমার", "মস্তিষ্কের টিউমার", "ब्रेन ट्यूमर", "brain tumour", "brain tumor", "brain surgery", "spine surgery"],
    },
    "Urology": {
        2: [
            "প্রস্রাব", "প্রস্রাবে", "প্রস্রাবের", "পেচ্ছাপ", "পেসাব", "প্রসাব", "প্রস্রাবে জ্বালা", "প্রস্রাবে রক্ত", "প্রস্রাব হচ্ছে না", "কিডনি", "কিডনিতে", "কিডনির", "প্রোস্টেট", "ইউটিআই",
            "ইউটি আই", "মূত্র", "মূত্রথলি", "ব্লাডার", "লিঙ্গ", "অণ্ডকোষ", "অন্ডকোষ", "সুন্নত", "খতনা", "ইউরিন",
            "पेशाब", "मूत्र", "किडनी", "प्रोस्टेट", "यूटीआई", "ब्लैडर", "यूरिन", "पथरी",
            "urine", "urinary", "kidney", "kidneys", "kidney stone", "prostate", "bladder", "uti", "urology", "urologist", "circumcision", "testicle", "pechhap", "pesab", "prosrab",
        ],
    },
    "Gastroenterology": {
        2: [
            "লিভার", "লিবার", "জন্ডিস", "জণ্ডিস", "কামলা", "হেপাটাইটিস", "ফ্যাটি লিভার", "আলসার", "গ্যাস্ট্রিক", "অ্যাসিডিটি", "বুক জ্বালা", "গ্যাস্ট্রো",
            "लिवर", "लीवर", "पीलिया", "जिगर", "गैस्ट्रिक", "एसिडिटी", "अल्सर", "हेपेटाइटिस",
            "liver", "jaundice", "hepatitis", "fatty liver", "gastric", "acidity", "ulcer", "heartburn", "indigestion", "gastroenterologist",
        ],
    },
    "Psychiatry": {
        2: [
            "ডিপ্রেশন", "ডিপ্রেসন", "বিষণ্ণতা", "বিষণ্নতা", "দুশ্চিন্তা", "উদ্বেগ", "অ্যাংজাইটি", "প্যানিক", "ঘুম আসে না", "ঘুম হয় না", "ঘুমের সমস্যা", "মানসিক", "মন খারাপ", "নেশা", "মদ খাই",
            "আত্মহত্যা", "পাগল", "সাইকিয়াট্রিস্ট", "মনোরোগ", "ওসিডি", "সিজোফ্রেনিয়া", "মনের অসুখ",
            "डिप्रेशन", "अवसाद", "चिंता", "घबराहट", "पैनिक", "नींद नहीं", "तनाव", "नशा", "मानसिक", "उदासी", "मन खराब", "पागल",
            "depression", "anxiety", "stress", "insomnia", "addiction", "mental", "suicidal", "psychiatrist", "sleepless", "panic",
        ],
    },
    "General Medicine": {
        1: [
            "জ্বর", "জ্বরে", "জ্বর জ্বর", "সর্দি", "কাশি", "কফ", "ঠান্ডা লাগা", "ঠান্ডা লেগেছে", "গা গরম", "দুর্বল", "দুর্বলতা", "বমি", "বমি বমি", "ডায়রিয়া", "পাতলা পায়খানা", "পেট খারাপ",
            "পেট ব্যথা", "পেটে ব্যথা", "শ্বাসকষ্ট", "শ্বাসের কষ্ট", "হাঁপানি", "ফুসফুস", "গা ব্যথা", "গা হাত পা ব্যথা", "শরীর ব্যথা", "অ্যালার্জি", "এলার্জি", "ডেঙ্গু", "ম্যালেরিয়া",
            "টাইফয়েড", "ডিহাইড্রেশন", "চেকআপ", "হেলথ চেকআপ", "অসুস্থ", "শরীর খারাপ", "শরীর ভালো নেই", "কেটে গেছে", "ক্ষত", "ইনফেকশন",
            "बुखार", "सर्दी", "खांसी", "खाँसी", "कमजोरी", "कमज़ोरी", "उल्टी", "दस्त", "पेट खराब", "पेट दर्द", "सांस", "साँस", "दमा", "बदन दर्द", "शरीर दर्द", "चेकअप", "तबीयत खराब", "एलर्जी",
            "fever", "cold", "cough", "weakness", "vomiting", "diarrhoea", "diarrhea", "loose motion", "breathless", "asthma", "body ache", "checkup", "check-up", "unwell", "sick", "allergy",
            "infection", "swelling", "stomach pain", "jor", "jwor", "sardi", "kashi",
        ],
        # (no "ache" -- in romanised Bengali it means "is" ("doctor ke ache?") -- and no bare "problem" / "সমস্যা": "what is the problem?" is a question the agent must still ask)
        0.5: ["ব্যথা", "ব্যাথা", "যন্ত্রণা", "দর্দ", "दर्द", "तकलीफ", "pain"],
    },
    "Ophthalmology": {
        2: [
            "চোখ", "চোখে", "চোখের", "চক্ষু", "চশমা", "ছানি", "ঝাপসা", "দৃষ্টি", "চোখ লাল", "চোখ জ্বালা", "চোখ দিয়ে জল",
            "आँख", "आंख", "आँखों", "चश्मा", "मोतियाबिंद", "धुंधला",
            "eye", "eyes", "eyesight", "vision", "cataract", "spectacles", "glasses", "chokh", "aankh",
        ],
    },
}

# ------------------------------------------------------------------------------------------------------------------------------ folding
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_FOLD = {
    # Bengali: long/short vowel signs, three sibilants, ণ, nukta forms, chandrabindu, ৎ
    "ী": "ি", "ূ": "ু", "ণ": "ন", "শ": "স", "ষ": "স", "ৎ": "ত", "ঁ": "", "়": "", "ঈ": "ই", "ঊ": "উ", "ঢ": "ঢ",
    # Devanagari
    "ी": "ि", "ू": "ु", "ँ": "ं", "़": "", "ण": "न", "श": "स", "ष": "स", "ई": "इ", "ऊ": "उ", "ऑ": "ओ", "ॉ": "ो", "ऍ": "ए",
}
_TOKEN = re.compile(r"[0-9A-Za-zऀ-ॿঀ-৿]+")  # letters AND the vowel signs / nukta / hasanta of Devanagari and Bengali (\w would cut a word at every vowel sign)
# endings a word takes in a sentence: ছুলি -> ছুলিতে / ছুলির / ছুলিটা; kan -> kane / kaner
_SUFFIXES = tuple(sorted({
    "", "ে", "র", "ের", "য়", "য", "তে", "কে", "ও", "ই", "টা", "টি", "টার", "গুলো", "গুলি", "রা", "দের", "ে", "েই", "েও", "েতে", "ির", "িতে", "ার", "াতে", "াকে",
    "e", "er", "ke", "ta", "ti", "te", "ra", "o", "i", "r", "ey", "ete", "ye", "yer", "ate", "ake",
    "ों", "ो", "ें", "े", "का", "की", "के", "में", "ने", "ों",
}, key=len, reverse=True))


def fold(text: str) -> str:
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = t.translate(_ZW)
    t = "".join(_FOLD.get(ch, ch) for ch in t)
    t = unicodedata.normalize("NFC", t)
    # doubled Latin letters ("khoss", "chulli") and the long Latin vowels ("aa", "ee", "oo") are the short ones
    t = re.sub(r"([a-z])\1+", r"\1", t)
    return t


def _words(text: str) -> list[str]:
    return _TOKEN.findall(fold(text))


def _edit1(a: str, b: str) -> bool:
    """True when `a` and `b` differ by at most one inserted, dropped or changed letter."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _word_meets(sentence_word: str, table_word: str) -> int:
    """2: the same word, or the same with an ordinary ending. 1: one letter away (a long word only). 0: not the word."""
    if sentence_word == table_word:
        return 2
    if len(table_word) >= 2 and sentence_word.startswith(table_word):
        if sentence_word[len(table_word):] in _SUFFIXES:
            return 2
    # a one-letter slip is trusted only in a long word (a short one is too easily another word)
    if len(table_word) >= 5 and len(sentence_word) >= 4 and _edit1(sentence_word, table_word):
        return 1
    if len(table_word) >= 5 and len(sentence_word) > len(table_word):
        stem = sentence_word[: len(table_word) + 1]
        if _edit1(stem, table_word) and sentence_word[len(table_word) + 1:] in _SUFFIXES:
            return 1
    return 0


def _phrase_hits(words: list[str], phrase_words: list[str]) -> int:
    """2 when the table words come in order (at most two sentence words between neighbours) and every one is an exact / ordinary-ending match; 1 when one of them is only a near match; 0 otherwise."""
    if not phrase_words:
        return 0
    best = 0
    for start, w in enumerate(words):
        q0 = _word_meets(w, phrase_words[0])
        if not q0:
            continue
        pos, quality, ok = start, q0, True
        for nxt in phrase_words[1:]:
            found = None
            for j in range(pos + 1, min(len(words), pos + 4)):
                q = _word_meets(words[j], nxt)
                if q:
                    found, quality = j, min(quality, q)
                    break
            if found is None:
                ok = False
                break
            pos = found
        if ok:
            best = max(best, quality)
    return best


_PREPARED: list[tuple[str, float, list[str]]] | None = None


def _prepared() -> list[tuple[str, float, list[str]]]:
    global _PREPARED
    if _PREPARED is None:
        rows = []
        for dept, by_weight in LEXICON.items():
            for weight, terms in by_weight.items():
                for term in dict.fromkeys(terms):
                    pw = _words(term)
                    if pw:
                        rows.append((dept, float(weight), pw))
        _PREPARED = rows
    return _PREPARED


def score(text: str) -> dict[str, tuple[float, int]]:
    """department -> (strongest weight, number of distinct table entries that hit) for every department the sentence points at."""
    words = _words(text)
    out: dict[str, tuple[float, int]] = {}
    if not words:
        return out
    for dept, weight, pw in _prepared():
        q = _phrase_hits(words, pw)
        if q:
            w = weight if q == 2 else weight - 0.25  # a near match never ties with an exact one (মাসিক "period" is one letter from মানসিক "mental")
            best, n = out.get(dept, (0.0, 0))
            out[dept] = (max(best, w), n + 1)
    return out


def departments_for(text: str) -> list[str]:
    """The department(s) the sentence points at, strongest first. One name when one department is clearly strongest; two names (the caller is asked which) when two score the same; [] when
    no symptom word was found. A department that is only a vague word (weight 0.5) is returned only when nothing stronger was said."""
    found = score(text)
    if not found:
        return []
    ranked = sorted(found.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    top = ranked[0][1]
    return [ranked[0][0]] + [d for d, s in ranked[1:] if s == top][:1]

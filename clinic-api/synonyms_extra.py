"""Five more ways to say each lab test, in Bengali, Hindi and English (owner's request, 2026-10-17: "a test might be called 5 different names").

A caller says "the sugar test after food", "PPBS", "খাওয়ার পরে সুগার", a recogniser writes the English words in Bengali or Devanagari letters, and the
clinic's own spelling of the test is the last thing either of them uses. These are the lay, clinical, abbreviated and loan-word forms, one list per language per
test, kept as a static table (no database migration, exactly as i18n_content.TEST_ALIASES_EN is). They are ADDED to the aliases the database and
i18n_content already hold; a form that is already there is not repeated.

RULE (checked by tests/test_synonyms_extra.py): a form belongs to ONE test. A form shared by two tests would make the fast path pick the first silently,
so generic words ("sugar test", "heart test", "blood test", "scan") are deliberately absent. The spelled-letter forms of codes (সি বি সি, টি এস এইচ ...)
are not listed here: clinic-api/spelled_forms.py generates them for every test code.
"""

from __future__ import annotations

EXTRA: dict[str, dict[str, list[str]]] = {
    "Complete Blood Count (CBC)": {
        "bn": ["সিবিসি পরীক্ষা", "ব্লাড সেল কাউন্ট", "টোটাল ব্লাড কাউন্ট", "ফুল ব্লাড প্রোফাইল", "কম্পিলিট ব্লাড কাউন্ট"],
        "hi": ["ब्लड सेल काउंट", "टोटल ब्लड काउंट", "खून की गिनती", "कंप्लीट ब्लड सेल काउंट", "सीबीसी जांच"],
        "en": ["cbc blood test", "complete blood cell count", "total blood count", "full blood profile", "blood cell count"],
    },
    "ESR": {
        "bn": ["ই এস আর টেস্ট", "রক্তের থিতানো পরীক্ষা", "এরিথ্রোসাইট সেডিমেন্টেশন রেট", "সেড রেট", "ইএসআর পরীক্ষা"],
        "hi": ["ईएसआर टेस्ट", "ईएसआर जांच", "एरिथ्रोसाइट सेडिमेंटेशन रेट", "सेड रेट", "खून के बैठने की जांच"],
        "en": ["esr test", "esr blood test", "erythrocyte sedimentation", "blood sedimentation rate", "sedimentation rate test"],
    },
    "Blood Sugar Fasting": {
        "bn": ["ফাস্টিং ব্লাড সুগার", "ফাস্টিং সুগার টেস্ট", "খালি পেটে ব্লাড সুগার", "না খেয়ে সুগার", "এফবিএস"],
        "hi": ["फास्टिंग ब्लड शुगर", "फास्टिंग शुगर टेस्ट", "खाली पेट ब्लड शुगर", "बिना खाए शुगर", "एफबीएस"],
        "en": ["fbs", "fbs test", "fasting glucose", "fasting blood glucose", "fasting plasma glucose"],
    },
    "Blood Sugar PP": {
        "bn": ["পোস্ট প্র্যান্ডিয়াল সুগার", "খাবারের পর সুগার", "খাওয়ার দুই ঘণ্টা পর সুগার", "পিপিবিএস", "পিপি ব্লাড সুগার"],
        "hi": ["पोस्ट प्रैंडियल शुगर", "खाने के दो घंटे बाद शुगर", "खाने के बाद ब्लड शुगर", "पीपीबीएस", "पीपी ब्लड शुगर"],
        "en": ["ppbs", "pp blood sugar", "post meal sugar", "sugar after lunch", "two hour post meal sugar"],
    },
    "HbA1c": {
        "bn": ["এইচবিএ ওয়ান সি", "হিমোগ্লোবিন এওয়ানসি", "তিন মাসের সুগার", "গড় সুগার পরীক্ষা", "গ্লাইকোসাইলেটেড হিমোগ্লোবিন"],
        "hi": ["एचबीए वन सी टेस्ट", "तीन महीने की शुगर", "औसत शुगर जांच", "ग्लाइकोसिलेटेड हीमोग्लोबिन", "ए वन सी"],
        "en": ["a1c", "a1c test", "hb a one c", "three month sugar test", "average sugar test"],
    },
    "Lipid Profile": {
        "bn": ["লিপিড টেস্ট", "কোলেস্টেরল পরীক্ষা", "চর্বির পরীক্ষা", "লিপিড প্যানেল", "কোলেস্টরল"],
        "hi": ["लिपिड टेस्ट", "कोलेस्ट्रॉल जांच", "चर्बी की जांच", "लिपिड पैनल", "कोलेस्ट्रोल"],
        "en": ["lipid panel", "lipid profile test", "cholesterol profile", "cholesterol panel", "fat profile"],
    },
    "Liver Function Test (LFT)": {
        "bn": ["লিভার ফাংশন", "লিভার প্রোফাইল", "লিভার পরীক্ষা", "লিভারের টেস্ট", "এলএফটি টেস্ট"],
        "hi": ["लिवर फंक्शन", "लिवर प्रोफाइल", "लिवर की जांच", "लिवर पैनल", "एलएफटी टेस्ट"],
        "en": ["lft test", "liver panel", "liver function", "liver enzyme test", "sgot sgpt test"],
    },
    "Kidney Function Test (KFT)": {
        "bn": ["কিডনি ফাংশন", "কিডনি প্রোফাইল", "কিডনির পরীক্ষা", "রেনাল ফাংশন টেস্ট", "কেএফটি টেস্ট"],
        "hi": ["किडनी फंक्शन", "किडनी प्रोफाइल", "किडनी की जांच", "रीनल फंक्शन टेस्ट", "केएफटी टेस्ट"],
        "en": ["kft test", "kidney function", "kidney profile", "renal profile", "rft test"],
    },
    "Thyroid Profile (T3 T4 TSH)": {
        "bn": ["থাইরয়েড ফাংশন টেস্ট", "থাইরয়েড প্যানেল", "টি থ্রি টি ফোর টিএসএইচ", "গলার গ্ল্যান্ডের পরীক্ষা", "থাইরয়েডের পরীক্ষা"],
        "hi": ["थायराइड फंक्शन टेस्ट", "थायराइड पैनल", "टी थ्री टी फोर टीएसएच", "गले की ग्रंथि की जांच", "थायराइड की जांच"],
        "en": ["thyroid function", "thyroid blood test", "thyroid hormone panel", "thyroid hormones test", "tft test"],
    },
    "TSH": {
        "bn": ["টিএসএইচ টেস্ট", "থাইরয়েড স্টিমুলেটিং হরমোন", "শুধু টিএসএইচ", "টিএসএইচ পরীক্ষা", "টি এস এইচ পরীক্ষা"],
        "hi": ["टीएसएच टेस्ट", "थायराइड स्टिमुलेटिंग हार्मोन", "सिर्फ टीएसएच", "टीएसएच जांच", "टी एस एच जांच"],
        "en": ["tsh test", "tsh blood test", "only tsh", "tsh level", "thyroid stimulating hormone test"],
    },
    "Urine Routine Examination": {
        "bn": ["ইউরিন আর ই", "প্রস্রাবের টেস্ট", "প্রস্রাব পরীক্ষা রুটিন", "ইউরিন রুটিন টেস্ট", "ইউরিন এক্সামিনেশন"],
        "hi": ["यूरिन टेस्ट रूटीन", "पेशाब की जांच", "पेशाब का टेस्ट", "यूरिन रूटीन जांच", "यूरिन एग्जामिनेशन"],
        "en": ["urine re", "urine r e", "urine routine test", "routine urine examination", "urine analysis"],
    },
    "Widal Test": {
        "bn": ["টাইফয়েডের পরীক্ষা", "উইডাল", "ওয়াইডাল", "টাইফয়েড জ্বরের টেস্ট", "টাইফয়েড পরীক্ষা"],
        "hi": ["टाइफाइड की जांच", "विडाल", "विडाल टेस्ट", "टाइफाइड बुखार का टेस्ट", "टाइफाइड जांच"],
        "en": ["widal test for typhoid", "widal blood test", "typhoid blood test", "typhoid check", "enteric fever test"],
    },
    "Dengue NS1 Antigen": {
        "bn": ["ডেঙ্গু এনএস ওয়ান", "ডেঙ্গু এনএস১ টেস্ট", "ডেঙ্গু অ্যান্টিজেন", "ডেঙ্গু জ্বরের টেস্ট", "ডেঙ্গু এন এস ওয়ান"],
        "hi": ["डेंगू एनएस वन", "डेंगू एनएस१ टेस्ट", "डेंगू एंटीजन", "डेंगू बुखार का टेस्ट", "डेंगू एन एस वन"],
        "en": ["ns1 test", "ns1 antigen", "dengue ns one", "dengue fever antigen", "dengue ns1 test"],
    },
    "Dengue IgG/IgM": {
        "bn": ["ডেঙ্গু অ্যান্টিবডি টেস্ট", "ডেঙ্গু আইজিজি আইজিএম", "ডেঙ্গু আই জি এম", "ডেঙ্গু অ্যান্টিবডি", "ডেঙ্গুর অ্যান্টিবডি পরীক্ষা"],
        "hi": ["डेंगू एंटीबॉडी टेस्ट", "डेंगू आईजीजी आईजीएम", "डेंगू आई जी एम", "डेंगू एंटीबॉडी", "डेंगू एंटीबॉडी जांच"],
        "en": ["dengue igg igm", "dengue antibodies", "dengue serology", "dengue igm test", "dengue igg test"],
    },
    "Malaria Antigen": {
        "bn": ["ম্যালেরিয়া জ্বরের টেস্ট", "ম্যালেরিয়া পরীক্ষা", "ম্যালেরিয়ার পরীক্ষা", "এমপি টেস্ট", "ম্যালেরিয়া পরজীবী টেস্ট"],
        "hi": ["मलेरिया बुखार का टेस्ट", "मलेरिया की जांच", "मलेरिया जांच", "एमपी टेस्ट", "मलेरिया एंटीजन टेस्ट"],
        "en": ["malaria fever test", "malaria parasite test", "mp test", "malaria blood test", "malaria rapid test"],
    },
    "CRP (C-Reactive Protein)": {
        "bn": ["সি রিঅ্যাকটিভ প্রোটিন", "সিআরপি টেস্ট", "সিআরপি পরীক্ষা", "প্রদাহের পরীক্ষা", "সি আর পি টেস্ট"],
        "hi": ["सी रिएक्टिव प्रोटीन", "सीआरपी टेस्ट", "सीआरपी जांच", "सूजन की जांच", "सी आर पी टेस्ट"],
        "en": ["crp", "c r p test", "inflammation test", "reactive protein test", "c-reactive protein test"],
    },
    "Vitamin D (25-OH)": {
        "bn": ["ভিটামিন ডি টেস্ট", "ভিটামিন ডি থ্রি", "ভিটামিন ডি পরীক্ষা", "পঁচিশ ওএইচ ভিটামিন ডি", "ভিটামিন ডি লেভেল"],
        "hi": ["विटामिन डी टेस्ट", "विटामिन डी थ्री", "विटामिन डी जांच", "पच्चीस ओएच विटामिन डी", "विटामिन डी लेवल"],
        "en": ["vitamin d3", "vitamin d level", "vitamin d 25 oh", "25 hydroxy vitamin d", "vit d test"],
    },
    "Vitamin B12": {
        "bn": ["ভিটামিন বি টুয়েলভ", "বি১২ টেস্ট", "বি টুয়েলভ", "ভিটামিন বি১২ পরীক্ষা", "ভিটামিন বি বারো"],
        "hi": ["विटामिन बी बारह", "बी१२ टेस्ट", "बी बारह", "विटामिन बी१२ जांच", "विटामिन बी ट्वेल्व"],
        "en": ["vitamin b twelve", "b twelve", "b12 level", "cobalamin test", "vitamin b12 level"],
    },
    "Serum Creatinine": {
        "bn": ["ক্রিয়েটিনিন টেস্ট", "সিরাম ক্রিয়েটিনিন", "ক্রিয়েটিনিন পরীক্ষা", "কিডনির ক্রিয়েটিনিন", "ক্রিয়াটিনিন"],
        "hi": ["क्रिएटिनिन टेस्ट", "सीरम क्रिएटिनिन", "क्रिएटिनिन जांच", "किडनी क्रिएटिनिन", "क्रियाटिनिन"],
        "en": ["serum creatinine test", "creatinine level", "blood creatinine", "s creatinine", "creatinine blood test"],
    },
    "Serum Electrolytes": {
        "bn": ["ইলেকট্রোলাইট টেস্ট", "সোডিয়াম পটাশিয়াম টেস্ট", "সিরাম ইলেকট্রোলাইট", "ইলেকট্রোলাইটস", "রক্তের লবণের পরীক্ষা"],
        "hi": ["इलेक्ट्रोलाइट टेस्ट", "सोडियम पोटैशियम टेस्ट", "सीरम इलेक्ट्रोलाइट", "इलेक्ट्रोलाइट्स", "खून में नमक की जांच"],
        "en": ["serum electrolyte test", "sodium test", "potassium test", "electrolyte panel", "na k test"],
    },
    "Blood Grouping & Rh Typing": {
        "bn": ["ব্লাড গ্রুপ টেস্ট", "রক্তের গ্রুপ", "ব্লাড গ্রুপিং", "রক্তের গ্রুপ পরীক্ষা", "আরএইচ টাইপিং"],
        "hi": ["ब्लड ग्रुप टेस्ट", "खून का ग्रुप", "ब्लड ग्रुपिंग", "खून का ग्रुप जांच", "आरएच टाइपिंग"],
        "en": ["blood group and rh", "abo and rh typing", "blood type", "blood group check", "rh factor test"],
    },
    "HIV Test (ELISA)": {
        "bn": ["এইচআইভি টেস্ট", "এইডস টেস্ট", "এইচ আই ভি টেস্ট", "এইডসের পরীক্ষা", "এইচআইভি পরীক্ষা"],
        "hi": ["एचआईवी टेस्ट", "एड्स टेस्ट", "एच आई वी टेस्ट", "एड्स की जांच", "एचआईवी जांच"],
        "en": ["hiv elisa", "hiv elisa test", "hiv screening", "hiv blood test", "aids check"],
    },
    "HBsAg": {
        "bn": ["এইচবিএসএজি", "হেপাটাইটিস বি টেস্ট", "হেপাটাইটিস বি পরীক্ষা", "হেপাটাইটিস বি অ্যান্টিজেন", "হেপ বি টেস্ট"],
        "hi": ["एचबीएसएजी", "हेपेटाइटिस बी टेस्ट", "हेपेटाइटिस बी जांच", "हेपेटाइटिस बी एंटीजन", "हेप बी टेस्ट"],
        "en": ["hbsag test", "hepatitis b surface antigen", "hepatitis b antigen", "hepatitis b screening", "australia antigen"],
    },
    "HCV": {
        "bn": ["হেপাটাইটিস সি টেস্ট", "হেপাটাইটিস সি পরীক্ষা", "এইচসিভি টেস্ট", "হেপ সি টেস্ট", "হেপাটাইটিস সি অ্যান্টিবডি"],
        "hi": ["हेपेटाइटिस सी टेस्ट", "हेपेटाइटिस सी जांच", "एचसीवी टेस्ट", "हेप सी टेस्ट", "हेपेटाइटिस सी एंटीबॉडी"],
        "en": ["hcv test", "hepatitis c antibody", "hepatitis c screening", "hep c test", "hcv antibody test"],
    },
    "ECG": {
        "bn": ["ইসিজি টেস্ট", "হার্টের ইসিজি", "ইলেক্ট্রোকার্ডিওগ্রাম", "ইকেজি", "হার্টের গতির পরীক্ষা"],
        "hi": ["ईसीजी टेस्ट", "दिल का ईसीजी", "इलेक्ट्रोकार्डियोग्राम", "ईकेजी", "दिल की धड़कन की जांच"],
        "en": ["ecg done", "heart ecg", "electrocardiograph", "ekg test", "heart rhythm test"],
    },
    "Chest X-Ray (PA view)": {
        "bn": ["চেস্ট এক্স রে", "বুকের এক্স রে", "বুকের এক্সরে", "চেস্ট এক্সরে", "ফুসফুসের এক্স রে"],
        "hi": ["चेस्ट एक्स रे", "छाती का एक्सरे", "सीने का एक्स रे", "चेस्ट एक्सरे", "फेफड़ों का एक्स रे"],
        "en": ["chest x ray", "x ray chest", "cxr", "chest radiograph", "x-ray of chest"],
    },
    "USG Whole Abdomen": {
        "bn": ["পেটের আল্ট্রাসাউন্ড", "পেটের সোনোগ্রাফি", "পেটের ইউএসজি", "ইউএসজি অ্যাবডোমেন", "পেটের আলট্রা সনোগ্রাফি"],
        "hi": ["पेट का अल्ट्रासाउंड", "पेट की सोनोग्राफी", "पेट का यूएसजी", "यूएसजी एब्डोमेन", "पेट की अल्ट्रा सोनोग्राफी"],
        "en": ["usg abdomen", "whole abdomen ultrasound", "abdominal ultrasound", "abdominal sonography", "usg whole abdomen test"],
    },
    "USG Pregnancy Profile": {
        "bn": ["গর্ভাবস্থার আল্ট্রাসাউন্ড", "প্রেগনেন্সি সোনোগ্রাফি", "গর্ভাবস্থার সোনোগ্রাফি", "প্রেগনেন্সি ইউএসজি", "বেবি স্ক্যান"],
        "hi": ["प्रेगनेंसी अल्ट्रासाउंड", "गर्भावस्था की सोनोग्राफी", "प्रेगनेंसी सोनोग्राफी", "प्रेगनेंसी यूएसजी", "बेबी स्कैन"],
        "en": ["pregnancy sonography", "obstetric ultrasound", "baby scan", "antenatal scan", "pregnancy profile usg"],
    },
    "2D Echocardiography": {
        "bn": ["ইকো টেস্ট", "টু ডি ইকো", "হার্টের আল্ট্রাসাউন্ড", "ইকোকার্ডিওগ্রাফি", "হার্টের ইকো"],
        "hi": ["इको टेस्ट", "टू डी इको", "दिल का अल्ट्रासाउंड", "इकोकार्डियोग्राफी", "दिल का इको"],
        "en": ["echo cardiography", "echocardiography", "two d echo", "heart echo", "cardiac echo"],
    },
    "TMT (Treadmill Test)": {
        "bn": ["ট্রেডমিল টেস্ট", "টিএমটি টেস্ট", "স্ট্রেস টেস্ট", "হাঁটার হার্ট টেস্ট", "ট্রেডমিলে হার্ট পরীক্ষা"],
        "hi": ["ट्रेडमिल टेस्ट", "टीएमटी टेस्ट", "स्ट्रेस टेस्ट", "चलकर दिल की जांच", "ट्रेडमिल पर दिल की जांच"],
        "en": ["tmt", "tread mill test", "exercise stress test", "exercise ecg", "treadmill stress test"],
    },
    "Pap Smear": {
        "bn": ["প্যাপ স্মিয়ার", "প্যাপ টেস্ট", "জরায়ুমুখের পরীক্ষা", "সার্ভিক্যাল স্ক্রিনিং", "প্যাপ স্মিয়ার টেস্ট"],
        "hi": ["पैप स्मियर", "पैप टेस्ट", "गर्भाशय ग्रीवा की जांच", "सर्वाइकल स्क्रीनिंग", "पैप स्मियर टेस्ट"],
        "en": ["pap smear", "papanicolaou test", "cervical screening", "cervical smear", "cervix cancer screening"],
    },
    "PSA (Prostate Specific Antigen)": {
        "bn": ["পিএসএ টেস্ট", "প্রোস্টেট টেস্ট", "প্রোস্টেট স্পেসিফিক অ্যান্টিজেন", "প্রোস্টেটের পরীক্ষা", "পিএসএ পরীক্ষা"],
        "hi": ["पीएसए टेस्ट", "प्रोस्टेट टेस्ट", "प्रोस्टेट स्पेसिफिक एंटीजन", "प्रोस्टेट की जांच", "पीएसए जांच"],
        "en": ["psa", "prostate specific antigen", "prostate blood test", "prostate cancer marker", "psa level"],
    },
    "Uric Acid": {
        "bn": ["ইউরিক অ্যাসিড টেস্ট", "ইউরিক এসিড", "ইউরিক অ্যাসিড পরীক্ষা", "গাউটের পরীক্ষা", "ইউরিক অ্যাসিডের টেস্ট"],
        "hi": ["यूरिक एसिड टेस्ट", "यूरिक एसिड", "यूरिक एसिड जांच", "गाउट की जांच", "यूरिक ऐसिड"],
        "en": ["uric acid", "uric acid test", "uric acid level", "serum uric acid", "gout test"],
    },
    "Calcium (Serum)": {
        "bn": ["ক্যালসিয়াম টেস্ট", "সিরাম ক্যালসিয়াম", "ক্যালসিয়াম পরীক্ষা", "রক্তের ক্যালসিয়াম", "ক্যালশিয়াম"],
        "hi": ["कैल्शियम टेस्ट", "सीरम कैल्शियम", "कैल्शियम जांच", "खून में कैल्शियम", "कैल्सियम"],
        "en": ["calcium level", "blood calcium", "calcium blood test", "s calcium", "calcium serum"],
    },
}


def extra(name: str, lang: str) -> list[str]:
    return list(EXTRA.get(name, {}).get(lang, []))

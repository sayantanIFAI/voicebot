"""Hindi and English content for the seeded catalogue.

Same status as the rest of seed.py: fictional clinic details, authored for
the prototype, not sourced from a real clinic. Kept in its own file so the
Bengali seed data stays readable and so `backfill_i18n()` can bring an
EXISTING database up to date without a destructive reseed (which would
wipe every appointment booked since the last boot).

Aliases are what a caller's ASR output actually looks like: Hindi ASR
emits Devanagari, including for English loan words ("यूरिक एसिड"), so the
Devanagari spelling is the lookup key, exactly as aliases_bn is for Bengali.
"""

from __future__ import annotations

TEST_ALIASES_HI: dict[str, list[str]] = {
    "Complete Blood Count (CBC)": ["सीबीसी", "सी बी सी", "कंप्लीट ब्लड प्रोफाइल", "कम्प्लीट ब्लड प्रोफाइल", "कंप्लीट ब्लड काउंट", "कम्प्लीट ब्लड काउंट", "कंप्लीट ब्लड पिक्चर", "फुल ब्लड काउंट", "ब्लड काउंट", "ब्लड प्रोफाइल", "हीमोग्राम", "सीबीसी टेस्ट", "सी बी सी टेस्ट"],
    "ESR": ["ईएसआर", "ई एस आर"],
    "Blood Sugar Fasting": ["ब्लड शुगर फास्टिंग", "शुगर फास्टिंग", "खाली पेट शुगर"],
    "Blood Sugar PP": ["शुगर पीपी", "पीपी शुगर", "खाने के बाद शुगर"],
    "HbA1c": ["एचबीए वन सी", "हीमोग्लोबिन ए वन सी"],
    "Lipid Profile": ["लिपिड प्रोफाइल", "कोलेस्ट्रॉल टेस्ट"],
    "Liver Function Test (LFT)": ["लिवर फंक्शन टेस्ट", "एलएफटी", "लिवर टेस्ट", "पीलिया का टेस्ट", "पीलिया टेस्ट"],
    "Kidney Function Test (KFT)": ["किडनी फंक्शन टेस्ट", "केएफटी", "किडनी टेस्ट"],
    "Thyroid Profile (T3 T4 TSH)": ["थायराइड प्रोफाइल", "थायराइड टेस्ट", "थायराइड", "थाइरॉइड", "थायरॉइड", "थाइराइड टेस्ट"],
    "TSH": ["टीएसएच"],
    "Urine Routine Examination": ["यूरिन टेस्ट", "पेशाब की जांच", "यूरिन रूटीन", "मूत्र का टेस्ट", "मूत्र टेस्ट", "मूत्र की जांच", "मूत्र जांच", "पेशाब का टेस्ट", "पेशाब टेस्ट", "पेशाब की जाँच", "यूरिन की जांच", "यूरिन का टेस्ट", "मूत्र परीक्षण"],
    "Widal Test": ["विडाल टेस्ट", "टाइफाइड टेस्ट"],
    "Dengue NS1 Antigen": ["डेंगू एनएस वन", "डेंगू टेस्ट"],
    "Dengue IgG/IgM": ["डेंगू आईजीजी", "डेंगू आईजीएम"],
    "Malaria Antigen": ["मलेरिया टेस्ट", "मलेरिया एंटीजन"],
    "CRP (C-Reactive Protein)": ["सीआरपी"],
    "Vitamin D (25-OH)": ["विटामिन डी"],
    "Vitamin B12": ["विटामिन बी बारह", "बी बारह"],
    "Serum Creatinine": ["क्रिएटिनिन", "सीरम क्रिएटिनिन"],
    "Serum Electrolytes": ["इलेक्ट्रोलाइट्स", "इलेक्ट्रोलाइट टेस्ट"],
    "Blood Grouping & Rh Typing": ["ब्लड ग्रुप", "खून का ग्रुप"],
    "HIV Test (ELISA)": ["एचआईवी टेस्ट", "एड्स टेस्ट"],
    "HBsAg": ["एचबीएसएजी", "हेपेटाइटिस बी"],
    "HCV": ["एचसीवी", "हेपेटाइटिस सी"],
    "ECG": ["ईसीजी", "इलेक्ट्रोकार्डियोग्राम"],
    "Chest X-Ray (PA view)": ["छाती का एक्स-रे", "चेस्ट एक्सरे", "छाती की फोटो", "छाती की तस्वीर"],
    # "यूएसजी" (bare USG) added to BOTH 2026-10-12, same reasoning as seed.py's matching Bengali fix: a caller
    # who says only "USG test price" with no qualifier was matching neither alias and falling through to a
    # phonetic near-miss (Uric Acid) instead of the two-tests-fit ambiguity path this clinic already has.
    "USG Whole Abdomen": ["पेट का अल्ट्रासाउंड", "होल एब्डोमेन यूएसजी", "पेट का यूएसजी", "यूएसजी"],
    "USG Pregnancy Profile": ["प्रेगनेंसी अल्ट्रासाउंड", "प्रेगनेंसी यूएसजी", "यूएसजी"],
    "2D Echocardiography": ["इको टेस्ट", "एकोकार्डियोग्राफी", "इकोकार्डियोग्राम", "दिल की फोटो", "दिल का अल्ट्रासाउंड"],
    "TMT (Treadmill Test)": ["टीएमटी", "ट्रेडमिल टेस्ट"],
    "Pap Smear": ["पैप स्मियर"],
    "PSA (Prostate Specific Antigen)": ["पीएसए"],
    "Uric Acid": ["यूरिक एसिड"],
    "Calcium (Serum)": ["कैल्शियम", "सीरम कैल्शियम"],
}

# sourav, 2026-10-14: English lay-person and clinical spellings of each test. Why: Bengali and Hindi had aliases for
# "typhoid test", "kidney test", "stomach scan" ..., English had NONE, so almost every English lay phrase fell through
# to the AI model (found by tests/test_lab_price_multilingual.py). Flow: this table -> clinic-api/main.py
# catalogue() ("aliases_en", read by agent/fast_path.py Catalogue like aliases_<lang> for any language) AND
# _alias_blob() (so /api/v1/tests/search resolves the same words for the multi-test total). Kept as a static table
# on purpose: no schema change and no migration on the live database. A form must belong to ONE test (a form in two
# tests would make the fast path pick the first silently) -- "sugar test"/"heart test" are deliberately absent.
TEST_ALIASES_EN: dict[str, list[str]] = {
    "Complete Blood Count (CBC)": ["complete blood count", "full blood count", "blood count", "complete blood picture", "complete blood profile", "full blood picture", "blood picture", "complete haemogram", "complete hemogram", "haemogram", "hemogram", "cbc test", "cbp"],
    "ESR": ["erythrocyte sedimentation rate", "sed rate"],
    "Blood Sugar Fasting": ["fasting blood sugar", "fasting sugar", "empty stomach sugar", "empty stomach sugar test", "sugar fasting"],
    "Blood Sugar PP": ["post prandial sugar", "postprandial blood sugar", "pp sugar", "sugar after meals", "sugar after food"],
    "HbA1c": ["hba1c", "hb a1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    "Lipid Profile": ["cholesterol test", "cholesterol", "lipid test"],
    "Liver Function Test (LFT)": ["liver function test", "liver test", "jaundice test", "liver profile"],
    "Kidney Function Test (KFT)": ["kidney function test", "kidney test", "renal function test"],
    "Thyroid Profile (T3 T4 TSH)": ["thyroid test", "thyroid profile", "thyroid", "thyroid function test", "thyroid panel", "t3 t4 tsh"],
    "TSH": ["thyroid stimulating hormone"],  # sourav: "tsh test" dropped; the fast path already strips "test"
    "Urine Routine Examination": ["urine test", "urine routine", "urine examination", "urine check", "pee test"],
    "Widal Test": ["typhoid test", "typhoid fever test", "widal"],
    "Dengue NS1 Antigen": ["dengue test", "dengue ns1", "dengue antigen"],
    "Dengue IgG/IgM": ["dengue igg", "dengue igm", "dengue antibody test"],
    "Malaria Antigen": ["malaria test", "malaria antigen test"],
    "CRP (C-Reactive Protein)": ["c reactive protein", "crp test"],
    "Vitamin D (25-OH)": ["vitamin d", "vitamin d test", "vit d"],
    "Vitamin B12": ["vitamin b12", "b12 test", "vit b12"],
    "Serum Creatinine": ["creatinine", "creatinine test"],
    "Serum Electrolytes": ["electrolytes", "electrolyte test", "sodium potassium test"],
    "Blood Grouping & Rh Typing": ["blood group", "blood group test", "blood grouping", "blood type test", "rh typing"],
    "HIV Test (ELISA)": ["hiv", "hiv test", "aids test", "aids"],
    "HBsAg": ["hepatitis b", "hepatitis b test", "hep b"],
    "HCV": ["hepatitis c", "hepatitis c test", "hep c"],
    "ECG": ["ecg test", "ekg", "heart tracing", "electrocardiogram"],
    "Chest X-Ray (PA view)": ["chest x-ray", "chest xray", "chest picture", "lung x-ray"],
    "USG Whole Abdomen": ["stomach scan", "stomach picture", "stomach test", "abdomen scan", "abdomen ultrasound", "stomach ultrasound", "belly scan"],
    "USG Pregnancy Profile": ["pregnancy scan", "pregnancy ultrasound", "pregnancy usg"],
    "2D Echocardiography": ["echo test", "2d echo", "heart ultrasound", "echocardiogram", "echo"],
    "TMT (Treadmill Test)": ["treadmill test", "tmt test", "stress test"],
    "Pap Smear": ["pap smear test", "pap test"],
    "PSA (Prostate Specific Antigen)": ["psa test", "prostate test"],
    "Calcium (Serum)": ["calcium", "calcium test", "serum calcium"],
}

SURNAME_HI: dict[str, list[str]] = {
    "Mukherjee": ["मुखर्जी"],
    "Sen": ["सेन"],
    "Ghosh": ["घोष"],
    "Chowdhury": ["चौधरी"],
    "Bhattacharya": ["भट्टाचार्य"],
    "Roy": ["रॉय", "राय"],
    "Banerjee": ["बनर्जी"],
    "Dutta": ["दत्ता"],
    "Chatterjee": ["चटर्जी"],
    "Basu": ["बसु"],
    "Mitra": ["मित्रा"],
    "Sengupta": ["सेनगुप्ता"],
    "Das": ["दास"],
    "Bose": ["बोस"],
    "Kar": ["कर"],
    "Nandi": ["नंदी"],
    "Pal": ["पाल"],
    "Halder": ["हालदार"],
    "Guha": ["गुहा"],
    "Chanda": ["चंदा"],
    "Saha": ["साहा"],
    "Dey": ["डे"],
    "Adhikari": ["अधिकारी"],
    "Bagchi": ["बागची"],
    "Biswas": ["बिस्वास"],
    "Majumder": ["मजूमदार"],
    "Mondal": ["मंडल"],
    "Ganguly": ["गांगुली"],
    "Sinha": ["सिन्हा"],
    "Ray": ["रे", "राय"],
    "Sarkar": ["सरकार"],
    "Chakraborty": ["चक्रवर्ती"],
    "Kundu": ["कुंडू"],
    "Mallick": ["मल्लिक"],
    "Pramanik": ["प्रमाणिक"],
    "Bhowmik": ["भौमिक"],
}

# test name -> (Hindi, English). Tests not listed get the default below.
PREP_I18N: dict[str, tuple[str, str]] = {
    "Blood Sugar Fasting": (
        "इस टेस्ट से पहले कम से कम आठ घंटे कुछ न खाएँ, सिर्फ़ पानी पी सकते हैं। सुबह खाली पेट आकर टेस्ट कराना बेहतर है।",
        "Do not eat anything for at least eight hours before this test; you may drink water. "
        "It is best to come in the morning on an empty stomach.",
    ),
    "Lipid Profile": (
        "इस टेस्ट से पहले दस से बारह घंटे उपवास रखना होगा, पानी के अलावा कुछ न लें। पिछली रात हल्का खाना खाना अच्छा रहता है।",
        "Fast for ten to twelve hours before this test, nothing except water. A light meal the night before is best.",
    ),
    "HbA1c": (
        "इस टेस्ट के लिए उपवास की ज़रूरत नहीं है, सामान्य खाने के बाद भी करा सकते हैं।",
        "No fasting is needed for this test; you can have it after your normal meals.",
    ),
    "Kidney Function Test (KFT)": (
        "इस टेस्ट से पहले छह से आठ घंटे उपवास रखने की सलाह दी जाती है।",
        "Fasting for six to eight hours is advised before this test.",
    ),
    "Liver Function Test (LFT)": (
        "इस टेस्ट से पहले आठ घंटे उपवास रखने की सलाह दी जाती है।",
        "Fasting for eight hours is advised before this test.",
    ),
    "USG Whole Abdomen": (
        "इस टेस्ट से पहले छह घंटे कुछ न खाएँ और पेशाब रोककर रखें, मूत्राशय भरा होना ज़रूरी है।",
        "Do not eat anything for six hours before this test, and hold your urine; the bladder needs to be full.",
    ),
    "USG Pregnancy Profile": (
        "इस टेस्ट से पहले ज़्यादा पानी पिएँ ताकि मूत्राशय भरा रहे, और टेस्ट से ठीक पहले पेशाब न करें।",
        "Drink plenty of water before this test to keep your bladder full, and do not urinate just before the test.",
    ),
    "TMT (Treadmill Test)": (
        "हल्के, आरामदायक कपड़े और जूते पहनकर आएँ। टेस्ट से दो घंटे पहले भारी खाना न खाना बेहतर है।",
        "Wear light, comfortable clothes and shoes. It is best not to eat a heavy meal "
        "in the two hours before the test.",
    ),
}

DEFAULT_PREP_HI = "इस टेस्ट के लिए किसी ख़ास तैयारी की ज़रूरत नहीं है, आप सामान्य रूप से आकर करा सकते हैं।"
DEFAULT_PREP_EN = "No special preparation is needed for this test; you can come in normally."

# topic -> (Hindi, English)
FAQ_I18N: dict[str, tuple[str, str]] = {
    "hours": (
        "हमारा क्लिनिक हर दिन सुबह आठ बजे से रात आठ बजे तक खुला रहता है, और रविवार को सुबह आठ बजे से दोपहर दो बजे तक।",
        "Our clinic is open every day from 8 AM to 8 PM, and on Sundays from 8 AM to 2 PM.",
    ),
    # OWNER'S INSTRUCTION, 2026-10-13: clinic address changed to this one, all three languages (see the
    # matching Bengali change in seed.py's FAQ seed row for "location").
    "location": (
        "हमारे क्लिनिक का पता ४४, सी आई टी रोड, लेडीज़ पार्क के पास, बेनियापुकुर, कोलकाता है।",
        "Our clinic is at 44, C I T Road, near Ladies Park, Beniapukur, Kolkata.",
    ),
    "payment_methods": (
        "यहाँ नकद, सभी प्रमुख कार्ड, और ऑनलाइन पेमेंट भी हो जाता है।",
        "We accept cash, all kinds of cards, and UPI.",
    ),
    "insurance": (
        "प्रमुख बीमा कंपनियों की कैशलेस सुविधा उपलब्ध है। अपने कार्ड का नाम बताइए, काउंटर से पुष्टि कर दी जाएगी।",
        "Cashless facility is available with major insurance companies. "
        "Tell us your card's name and the counter will confirm it for you.",
    ),
    "parking": (
        "क्लिनिक की अपनी पार्किंग है, कोई शुल्क नहीं लगता।",
        "The clinic has its own parking, and there is no charge.",
    ),
    "report_collection": (
        "आप रिपोर्ट सीधे लैब काउंटर से प्राप्त कर सकते हैं, या हम इसे आपके व्हाट्सएप और ईमेल पर भी भेज सकते हैं।",
        "You can collect your report directly from the counter, or we can also send it on WhatsApp and email.",
    ),
    "contact_number": (
        "आप हमारे हेल्पडेस्क नंबर पर फ़ोन करके सीधे बात कर सकते हैं, इस कॉल के बाद नंबर एसएमएस से भी भेज दिया जाएगा।",
        "You can call our helpdesk number and speak to us directly. We will also SMS you the number after this call.",
    ),
    "home_collection": (
        "ज़्यादातर ब्लड टेस्ट के लिए घर पर ही सैंपल कलेक्शन की सुविधा उपलब्ध है। कृपया बुकिंग करते समय इसकी जानकारी दे दें।",
        "Home sample collection is available for most blood tests. Please mention it when you book.",
    ),
}


# topic -> (Hindi keyword phrases, English keyword phrases). What FastPath matches a caller's words against for
# a clinic FAQ question (KCD-095). Phrases, not single generic words: a fast-path hit is answered without a
# model, so "time" alone would send a question about report time to the opening hours. REASONED, not measured
# against real calls; a native reviewer should add the phrasings callers actually use.
FAQ_KEYWORDS_I18N: dict[str, tuple[list[str], list[str]]] = {
    "hours": (
        [
            "क्लिनिक कब खुलता है",
            "क्लिनिक कब बंद होता है",
            "खुलने का समय",
            "बंद होने का समय",
            "क्लिनिक का समय",
            # OWNER'S INSTRUCTION, 2026-10-12: "test karwane kitne baje aau?" ("what time should I come to do
            # the test?") -- the answer is simply the clinic's own hours, already here; just needed the
            # phrasing to be recognised. Same reasoning applies to any task ("sample dene", "report lene") a
            # caller ties to a time-of-day question -- the clinic's hours answer it the same way.
            "टेस्ट करवाने कितने बजे आऊं",
            "टेस्ट करवाने कब आना होगा",
            "किस समय आना होगा",
            "कितने बजे आना होगा",
            "कब आना होगा",
        ],
        [
            "opening hours",
            "opening time",
            "closing time",
            "what time do you open",
            "what time do you close",
            "clinic timings",
            "clinic hours",
            "are you open",
            "what time should i come",
            "what time to come for the test",
            "when should i come for the test",
            "what time do i need to come",
        ],
    ),
    "location": (
        ["आप कहाँ हैं", "आप कहां हैं", "क्लिनिक कहाँ है", "क्लिनिक कहां है", "आपका पता", "पता क्या है", "कैसे पहुँचें", "कैसे पहुंचें"],
        [
            "where are you",
            "where is the clinic",
            "clinic address",
            "your address",
            "how do i get there",
            "how to reach",
            "where are you located",
            "where is it",
            "where is it located",
            "where is it situated",
            "where is it exactly",
            "where is the address",
            "what is the address",
            "what is the location",
            "location please",
            "where to go",
            "where to go for your clinic",
            "where is this place",
            "where are you situated",
            "location please",
            "where is your clinic",
            "where is your clinic located",
        ],
    ),
    "payment_methods": (
        [
            "पेमेंट कैसे",
            "भुगतान कैसे",
            "कार्ड चलता है",
            "यूपीआई",
            "नकद",
            # Colloquial loanword for cash, kept as its OWN short form so it
            # has a dedicated cue to win on, rather than falling through to
            # "insurance"'s "कैशलेस" (cashless) on character overlap alone --
            # "कैशलेस" literally starts with "कैश", so a bare "कैश" utterance
            # scored closer to "cashless" than to anything in this topic's
            # own list, which had no form containing that word at all.
            "कैश",
            "गूगल पे",
            "पेटीएम",
            "फोन पे",
            # sourav, 2026-10-14 (real call): "Online mein G Pay hoga" matched NO payment keyword and went to the
            # model, which answered with a greeting. The common ways of saying "I will pay online / by GPay".
            "जी पे",
            "जीपे",
            "फोनपे",
            "ऑनलाइन पेमेंट",
            "ऑनलाइन पेमेंट हो जाएगा",
            "पेमेंट ऑनलाइन",
            "ऑनलाइन में जी पे",
            "ऑनलाइन में गूगल पे",
            "ऑनलाइन पैसे",
            # "पेमेंट ऑप्शन क्या है" was committing to "location" instead (0.78 vs the 0.72 floor) on the shared
            # "क्या है" ("what is") tail alone, overlapping location's "पता क्या है" -- these own forms outscore
            # that collision directly. "पाय्मेंट" is kept as its own form too: the live ASR transcript spelled
            # "payment" that way (an extra य vowel sign), not as "पेमेंट", so the correctly-spelled form alone
            # never matched what the recogniser actually produced.
            "पेमेंट ऑप्शन क्या है",
            "पाय्मेंट ऑप्शन क्या है",
            "पाय्मेंट ऑप्शन",
            "पेमेंट ऑप्शन",
        ],
        [
            "payment methods",
            "pay by card",
            "do you take cards",
            "do you accept upi",
            "can i pay in cash",
            "how can i pay",
            "payment options",
            # Same reasoning as the Hindi "कैश" above -- a dedicated short
            # form for the bare word, not just inside a longer phrase.
            "cash",
            "google pay",
            "paytm",
            "phonepe",
            "phone pe",
            # sourav, 2026-10-14 (real call): the ASR wrote the Hindi sentence in English letters
            # ("Online Mein G Pay will be done") and no keyword matched. Phrases, not the bare word "online".
            "gpay",
            "g pay",
            "online payment",
            "pay online",
            "payment online",
            "payment will be made online",
            "online mein g pay",
            "net banking",
            "debit card",
            "credit card",
            "card payment",
        ],
    ),
    "insurance": (
        ["इंश्योरेंस", "बीमा", "कैशलेस", "मेडिक्लेम"],
        ["insurance", "cashless", "mediclaim", "health insurance"],
    ),
    "parking": (
        [
            "पार्किंग",
            "गाड़ी कहाँ खड़ी",
            "गाड़ी कहां खड़ी",
            # "गाड़ी" (car) and "जगह"/"जायेगा" are near-identical to their Bengali equivalents ("গাড়ি", "জায়গা") --
            # a Bengali caller asking this exact question is a live risk of the LANGUAGE detector itself
            # picking Hindi for the audio (a separate, known issue), so these forms need their own coverage in
            # the HINDI table too, not just Bengali's -- the utterance never reaches Bengali's cues at all when
            # that happens. Covers both the standard phrasing and what live ASR actually produced once.
            "गाड़ी रखने की जगह",
            "गाड़ी रखने के लिए जगह है",
            "गाड़ी रखा जा सकता है",
            "गाड़ी रखने को मिलेगा",
            "बाइक रखा जा सकता है",
            "बाइक रखने की जगह",
            "बाइक पार्किंग",
            "दोपहिया पार्किंग",
        ],
        [
            "parking",
            "car parking",
            "where can i park",
            "park my car",
            "bike parking",
            "is there parking",
            "do you have parking",
            "parking available",
            "parking space",
            "two wheeler parking",
            "motorcycle parking",
            "where do i park",
        ],
    ),
    "report_collection": (
        [
            "रिपोर्ट कैसे मिलेगी",
            "रिपोर्ट कहाँ से",
            "रिपोर्ट कहां से",
            "रिपोर्ट लेने",
            "रिपोर्ट कब मिलेगी",
            "रिपोर्ट कहा से मिलेगी",
            "रिपोर्ट दे दो",
            "रिपोर्ट चाहिए",
            "रिपोर्ट भेजो",
            "रिपोर्ट कहा है",
            "रिपोर्ट कैसे देखे",
            "रिपोर्ट डाउनलोड",
            "रिपोर्ट कैसे डाउनलोड करें",
            "रिपोर्ट व्हाट्सएप पर भेज दो",
            "रिपोर्ट लेने कब आऊ",
            "मेरी रिपोर्ट कहा है",
        ],
        [
            "collect my report",
            "report collection",
            "how do i get my report",
            "where to collect the report",
            "get the report",
            "report please",
            "need my report",
            "send my report",
            "where report",
            "give report",
            "report status",
            "where is my report",
            "how to view report",
            "download report",
            "report download",
            "send report pdf",
            "get report on whatsapp",
            "when will I get report",
        ],
    ),
    "contact_number": (
        ["फोन नंबर", "संपर्क नंबर", "हेल्पलाइन नंबर", "नंबर क्या है"],
        ["phone number", "contact number", "helpline number", "how can i contact you", "your number"],
    ),
    "home_collection": (
        [
            "होम कलेक्शन",
            "घर से सैंपल",
            "घर आकर",
            "घर पर सैंपल",
            "घर से ब्लड टेस्ट",
            "घर पर सैंपल कलेक्शन",
            "घर आ जाओ",
            "घर पर आकर ले लो",
            "सैंपल घर से",
            "घर पर टेस्ट",
            "घर से रिपोर्ट कलेक्शन",
            "घर पर सुविधा है",
            "ब्लड टेस्ट घर पर",
        ],
        [
            "home collection",
            "sample from home",
            "collect at home",
            "come to my house",
            "home visit for sample",
            "home test",
            "blood test at home",
            "home sample",
            "come home",
            "home collection available",
            "collect sample home",
            "home service",
            "send person home",
            "doorstep sample",
        ],
    ),
}


# ---------------------------------------------------------------------------------------------------------------------
# sourav, 2026-10-14: LOCATION / DIRECTION questions, in every accent the caller may use.
# WHY: callers ask "where / how do I reach you" in Bengali, Hindi, English, mixed (Hinglish/Banglish) and badly spelt
# Latin script, and the speech recogniser may write any of those whatever language the call is locked to. The fast path
# matches a sentence only against the keyword list of the CALL'S language, so a Latin-script or mixed sentence in a Bengali
# or Hindi call matched nothing and went to the model. Fix: the same romanised/mixed list is added to ALL THREE languages'
# "location" keywords, plus native-script sentences per language. Only keywords are added (backfill_faq_keywords appends
# them to the live DB on boot); the spoken answer (the clinic address) is unchanged.
# FLOW: caller asks -> fast path matches "location" keyword -> clinic API faq answer (address) in the call's language.
LOCATION_ROMAN: list[str] = [
    # Bengali, romanised
    "apnader location ta kothai", "apnader clinic ta kothay", "okhane kivabe jabo", "okhane ki bhabe pouchabo",
    "apnader address ta ektu bolben", "rasta ta kothay bolte parben", "apnader hospital ta kothai obosthito",
    "apnader center ta kothai obosthito", "city centre 2 er kache kothai", "kothay aste hobe amake",
    "kon sthan e apnader clinic", "kothay aste hobe", "clinic kothay", "address ta kothai", "location ta kothai",
    "kibhave jabo", "kibhave jabo waha par", "kaese jayenge okhane", "kivabe jabo", "kothay apnader clinic",
    # Hindi, romanised
    "aapka location kaha hai", "location kahan hai aapka", "waha kaise jayenge", "aapke clinic kaise pahunche",
    "waha ka address kya hai", "address batayiye na", "waha jaane ka rasta kya hai", "aapka center kahan par hai",
    "city centre 2 ke pas kahan hai", "kidhar aana padega humko", "kidhar aana padega", "clinic kaha hai",
    "location kaha hai", "address kya hai", "kaise pahunche", "kaise pahuchenge",
    # English
    "where is your location", "what is your clinic address", "how can i reach your center", "how to get to your location",
    "where are you located", "can you give me directions to your clinic", "how do i reach there", "what is the exact location",
    "where is the nearest landmark", "send me the location address", "directions to your clinic", "how to reach your center",
    "how do i get to the clinic", "where is your clinic", "give me the address", "location address",
    # mixed / code-switching
    "location kaha hai apnader", "address ta kidhar hai", "waha ka address ya location bolben",
    "acha ye batao apka clinic kaha par hai", "how to reach apnader center", "address ta kya hai apka",
    "apnader location kahan hai batao", "where is your clinic ta", "apnader address kahan par padega",
    "waha par kaise pahuchbo", "waha par kaise pahuchbo", "apka clinic kaha par hai", "clinic kaha par hai",
    # typos / phonetic / noisy
    "adress ta kothai", "kayse jayneg waha pr", "waha ka dress ya location", "acha hai kaha par ye apka clicnin",
    "locaton ka hai", "adress batao", "kase jau waha", "hot to reach center", "clnic kothaye", "adrez kya hai",
    "adress", "adrez", "locaton", "clnic kothaye", "kayse jayneg", "kase jau waha",
]
LOCATION_BN_NATIVE: list[str] = [
    "আপনাদের লোকেশন কোথায়", "আপনাদের ক্লিনিক কোথায়", "ওখানে কীভাবে যাব", "ওখানে কিভাবে যাব", "ওখানে কীভাবে পৌঁছাব",
    "ওখানে কিভাবে পৌঁছাব", "আপনাদের ঠিকানাটা একটু বলবেন", "ঠিকানাটা বলুন", "রাস্তাটা কোথায় বলতে পারবেন",
    "আপনাদের হাসপাতাল কোথায় অবস্থিত", "সিটি সেন্টার টু এর কাছে কোথায়", "কোথায় আসতে হবে আমাকে",
    "কোন স্থানে আপনাদের ক্লিনিক", "ক্লিনিক কোথায়", "ঠিকানা কী",
]
LOCATION_HI_NATIVE: list[str] = [
    "आपका लोकेशन कहाँ है", "लोकेशन कहाँ है आपका", "वहाँ कैसे जाएँगे", "वहां कैसे जाएंगे", "आपके क्लिनिक कैसे पहुँचे",
    "आपके क्लिनिक कैसे पहुंचे", "वहाँ का पता क्या है", "वहां का पता क्या है", "पता बताइए ना", "पता बताइए", "वहाँ जाने का रास्ता क्या है",
    "वहां जाने का रास्ता क्या है", "आपका सेंटर कहाँ पर है", "आपका सेंटर कहां पर है", "सिटी सेंटर दो के पास कहाँ है",
    "किधर आना पड़ेगा हमको", "किधर आना पड़ेगा", "लोकेशन बताइए",
]
LOCATION_EN_EXTRA: list[str] = [
    "where is your location", "what is your clinic address", "how can i reach your center", "how to get to your location",
    "can you give me directions to your clinic", "how do i reach there", "what is the exact location",
    "where is the nearest landmark", "send me the location address", "directions to the clinic",
    "tell me the way", "how far is it", "landmark near you", "get directions", "how to get there", "route to the clinic",
]
_loc_hi, _loc_en = FAQ_KEYWORDS_I18N["location"]
FAQ_KEYWORDS_I18N["location"] = (
    list(dict.fromkeys(_loc_hi + LOCATION_HI_NATIVE + LOCATION_ROMAN)),
    list(dict.fromkeys(_loc_en + LOCATION_EN_EXTRA + LOCATION_ROMAN)),
)

# sourav, 2026-10-14 (real call): a Bengali caller's "kibhabe jabo okhane" was written by the recogniser in DEVANAGARI
# ("की भाभी जाबो खाने", "उखाने जाबो की भावे") because the call's language was heard as Hindi, so neither the Hindi nor the Bengali
# keyword list matched it and the caller got "sorry, say again" twice. Bengali spoken-but-written-in-Devanagari and Hindi
# spoken-but-written-in-Bengali-script forms are added to ALL THREE "location" lists (the same reasoning as LOCATION_ROMAN).
LOCATION_CROSS_SCRIPT: list[str] = [
    # Bengali words, Devanagari spelling (several spellings: the recogniser is not consistent)
    "ओखाने कीभाबे जाबो", "ओखाने कि भाबे जाबो", "उखाने जाबो की भावे", "उखाने जाबो कीभावे", "की भाभी जाबो खाने", "की भाबे जाबो ओखाने",
    "कीभावे जाबो", "किभाबे जाबो", "ओखाने कीभाबे पौछाबो", "आपनादेर लोकेशन कोथाय", "आपनादेर क्लिनिक कोथाय", "आपनादेर एड्रेस कोथाय",
    "ठिकाना कोथाय", "क्लिनिक कोथाय", "कोथाय आसते होबे", "रास्ता कोथाय", "एड्रेस टा कोथाय", "लोकेशन टा कोथाय", "आपनादेर ठिकानाटा एकटू बोलबेन",
    # Hindi words, Bengali-script spelling
    "আপকা লোকেশন কাহা হ্যায়", "লোকেশন কাহান হ্যায় আপকা", "ওয়াহা কৈসে জায়েঙ্গে", "ওহা কইসে জায়েঙ্গে", "আপকে ক্লিনিক কৈসে পহুঁচে",
    "ওয়াহা কা অ্যাড্রেস ক্যা হ্যায়", "অ্যাড্রেস বতাইয়ে", "ওয়াহা জানে কা রাস্তা ক্যা হ্যায়", "কিধর আনা পড়েগা", "ক্লিনিক কাহা হ্যায়",
    "অ্যাড্রেস কাহা হ্যায়", "কৈসে পহুঁচেঙ্গে",
]
_h, _e = FAQ_KEYWORDS_I18N["location"]
# sourav, 2026-10-14 (real call, language lock = English): the Bengali sentence "ওখানে কিভাবে যাব" was matched only against
# the English list. Native Bengali and Hindi location sentences therefore go into every language's list as well.
FAQ_KEYWORDS_I18N["location"] = (
    list(dict.fromkeys(_h + LOCATION_CROSS_SCRIPT + LOCATION_BN_NATIVE)),
    list(dict.fromkeys(_e + LOCATION_CROSS_SCRIPT + LOCATION_BN_NATIVE + LOCATION_HI_NATIVE)),
)


# Owner, 2026-10-26 (real calls, Bengali and Hindi): the opening-hours and payment questions were said in ways no stored phrase covered -- "কটা থেকে খুলবে",
# "कितने बजे से खुलता है", "गूगल पाय फोन पाय सब हो जाएगा" -- and got "sorry, say again" / "जी, बताइए". Callers mix the languages, and the recogniser writes
# what it hears in whichever script it likes, so every phrase below goes into ALL THREE language lists (the same reasoning as LOCATION_CROSS_SCRIPT).
HOURS_BN_EXTRA: list[str] = [
    "কটা থেকে খুলবে", "কটা থেকে খোলে", "কটায় খোলে", "কটায় খুলবে", "কখন খুলবে", "কখন খুলে", "কটা পর্যন্ত খোলা", "কটা পর্যন্ত খোলা থাকে",
    "কটা থেকে কটা খুলবে", "কটা থেকে কটা পর্যন্ত", "কতক্ষণ খোলা", "ক্লিনিক কটা থেকে খোলে", "খোলার সময়", "বন্ধ হয় কটায়", "কটায় বন্ধ হয়",
    "কটার সময় আসবো", "কটার সময় আসব", "কখন আসবো কটা থেকে", "কখন থেকে খোলা", "কখন পর্যন্ত খোলা", "ক্লিনিক কখন খোলা", "কটা থেকে ক্লিনিক খোলে",
]
HOURS_HI_EXTRA: list[str] = [
    "कितने बजे से खुलता है", "कितने बजे खुलता है", "कितने बजे से खुलते हैं", "कब से खुलता है", "कब खुलता है", "कब खुलेगा", "कितने बजे तक खुला",
    "कितने बजे तक खुला रहता है", "कितने बजे तक", "कब बंद होता है", "कितने बजे बंद होता है", "खुलने का टाइम", "क्लिनिक का टाइम", "क्लिनिक कितने बजे",
    "कितने बजे से कितने बजे तक", "टाइमिंग क्या है", "क्लिनिक कब तक खुला", "कितने बजे आना है", "कितने बजे आऊं", "कितने बजे आऊँ",
]
PAYMENT_BN_EXTRA: list[str] = [
    "গুগল পে", "গুগল পেয়", "গুগল পে-তে", "ফোন পে", "ফোন পেয়", "ফোনপেয়", "পেটিএম", "পেটিম", "অনলাইনে দিলে", "অনলাইনে পেমেন্ট", "অনলাইনে দেব",
    "অনলাইন চলবে", "অনলাইনে হবে", "অনলাইনে দিতে পারব", "ক্যাশ ছাড়া", "কার্ডে পেমেন্ট", "কার্ডে হবে", "ইউ পি আই", "টাকা অনলাইনে", "অনলাইনে টাকা",
]
PAYMENT_HI_EXTRA: list[str] = [
    "गूगल पे", "गूगल पाय", "गूगल पेय", "फोन पे", "फोन पाय", "फोन पेय", "फोनपाय", "पेटीएम", "पेटिएम", "यूपीआई", "यू पी आई", "ऑनलाइन चलेगा", "ऑनलाइन देंगे",
    "ऑनलाइन दे सकते", "ऑनलाइन हो जाएगा", "कैश नहीं है", "कार्ड से पेमेंट", "कार्ड चलेगा", "पैसे ऑनलाइन", "ऑनलाइन पैसा", "पेमेंट करेंगे", "पाय्मेंट करेंगे",
    "पैसा ले लीजिएगा", "पैसे देंगे तो चलेगा", "गूगल पे से पैसा", "फोन पे से पैसा", "गूगल पाय से", "फोन पाय से",
]
PAYMENT_EN_EXTRA: list[str] = [
    "google pay", "gpay", "phonepe", "phone pay", "paytm", "pay online", "online payment", "will pay online", "upi payment", "can i pay online",
]
for _topic, _bn, _hi, _en in (
    ("hours", HOURS_BN_EXTRA, HOURS_HI_EXTRA, []),
    ("payment_methods", PAYMENT_BN_EXTRA, PAYMENT_HI_EXTRA, PAYMENT_EN_EXTRA),
):
    _h, _e = FAQ_KEYWORDS_I18N[_topic]
    FAQ_KEYWORDS_I18N[_topic] = (
        list(dict.fromkeys(_h + _hi + _bn + _en)),
        list(dict.fromkeys(_e + _hi + _bn + _en)),
    )

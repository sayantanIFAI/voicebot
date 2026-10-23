"""West Bengal administrative geography -- the seed rows of the four geo_* tables (owner, 2026-10-23, item 19).

Source: "West Bengal Administrative Relational Database.pdf" (a 3NF dataset of seven districts: Malda, Uttar Dinajpur, Dakshin Dinajpur, Kolkata, Howrah, Hooghly,
Murshidabad). Its ids are kept as the `code` of each row (DIST_01, PS_0101, SB_0101, PL_0101) so the PDF and the database can be compared line by line; the rows
point at each other through those codes, and models.py turns them into real foreign keys. The tables are DISTRIBUTED, not one master:

    geo_districts            1 --- N   geo_police_stations           1 --- N   geo_places
    geo_districts            1 --- N   geo_subdivisions_blocks       geo_districts 1 --- N geo_places

The Bengali and Hindi spellings (and a few Latin variants) are added so a recogniser that writes a place in another script still finds it; they are SPOKEN-FORM aliases
for matching only. The agent never says one of these names back: it only stores the table's own written name (clinic-api/geo_service.py).

THREE PLACES WHERE THE PDF DISAGREES WITH ITSELF (kept visible, not silently guessed):
  * the Kasba police-station row is typed "PS0406 / DIST04" (no underscores): read as PS_0406 / DIST_04, which every other row of the PDF points to;
  * the place "Gariahat" points at PS_0407, a police station the PDF never lists: stored with NO police station rather than invent one;
  * the place "Buniadpur" points at PS_0302 (Gangarampur PS) while a Buniadpur PS (PS_0303) exists in the same PDF: stored under PS_0303.
"""

from __future__ import annotations

# code, name, division, aliases_en, aliases_bn, aliases_hi   ("|"-joined)
DISTRICTS = [
    ("DIST_01", "Malda", "Malda", "Maldah|English Bazar Malda", "মালদা|মালদহ", "मालदा|मालदह"),
    ("DIST_02", "Uttar Dinajpur", "Malda", "North Dinajpur|Dinajpur Uttar", "উত্তর দিনাজপুর", "उत्तर दिनाजपुर"),
    ("DIST_03", "Dakshin Dinajpur", "Malda", "South Dinajpur|Dinajpur Dakshin", "দক্ষিণ দিনাজপুর", "दक्षिण दिनाजपुर"),
    ("DIST_04", "Kolkata", "Presidency", "Calcutta|Kolikata", "কলকাতা|কলিকাতা", "कोलकाता|कलकत्ता"),
    ("DIST_05", "Howrah", "Presidency", "Haora", "হাওড়া", "हावड़ा|हावरा"),
    ("DIST_06", "Hooghly", "Burdwan", "Hugli|Hoogly|Hooghli", "হুগলি|হুগলী", "हुगली"),
    ("DIST_07", "Murshidabad", "Malda", "Murshidabaad", "মুর্শিদাবাদ", "मुर्शिदाबाद"),
]

# code, district code, name (as written in the PDF), aliases_en, aliases_bn, aliases_hi
POLICE_STATIONS = [
    (
        "PS_0101",
        "DIST_01",
        "English Bazar PS",
        "Englishbazar|Inglish Bazar",
        "ইংলিশ বাজার|ইংরেজ বাজার",
        "इंग्लिश बाज़ार|इंग्लिश बाजार",
    ),
    ("PS_0102", "DIST_01", "Old Malda PS", "Purano Malda", "ওল্ড মালদা|পুরাতন মালদা", "ओल्ड मालदा"),
    ("PS_0103", "DIST_01", "Gazole PS", "Gajol", "গাজোল", "गाजोल"),
    ("PS_0104", "DIST_01", "Kaliachak PS", "Kaliyachak", "কালিয়াচক", "कालियाचक"),
    ("PS_0105", "DIST_01", "Chanchal PS", "Chanchol", "চাঁচল|চাঞ্চল", "चांचल|चाँचल"),
    ("PS_0201", "DIST_02", "Raiganj PS", "Raigunj", "রায়গঞ্জ", "रायगंज"),
    ("PS_0202", "DIST_02", "Islampur PS", "", "ইসলামপুর", "इस्लामपुर"),
    ("PS_0203", "DIST_02", "Dalkhola PS", "Dalkola", "ডালখোলা", "डालखोला"),
    ("PS_0204", "DIST_02", "Kaliaganj PS", "Kaliyaganj", "কালিয়াগঞ্জ", "कालियागंज"),
    ("PS_0301", "DIST_03", "Balurghat PS", "Balurghat", "বালুরঘাট", "बालुरघाट"),
    ("PS_0302", "DIST_03", "Gangarampur PS", "Gongarampur", "গঙ্গারামপুর", "गंगारामपुर"),
    ("PS_0303", "DIST_03", "Buniadpur PS", "Buniyadpur", "বুনিয়াদপুর", "बुनियादपुर"),
    ("PS_0401", "DIST_04", "Park Street PS", "Park Strit", "পার্ক স্ট্রিট", "पार्क स्ट्रीट"),
    ("PS_0402", "DIST_04", "New Market PS", "Newmarket", "নিউ মার্কেট", "न्यू मार्केट"),
    ("PS_0403", "DIST_04", "Alipore PS", "Alipur", "আলিপুর", "अलीपुर"),
    ("PS_0404", "DIST_04", "Shyampukur PS", "Shampukur|Syampukur", "শ্যামপুকুর", "श्यामपुकुर"),
    (
        "PS_0405",
        "DIST_04",
        "Bidhannagar / Salt Lake PS (Adj.)",
        "Bidhan Nagar|Saltlake|Salt Lake",
        "বিধাননগর|সল্টলেক|সল্ট লেক",
        "बिधाननगर|सॉल्ट लेक|साल्ट लेक",
    ),
    ("PS_0406", "DIST_04", "Kasba PS", "Kosba", "কসবা", "कसबा"),
    ("PS_0501", "DIST_05", "Howrah PS", "", "হাওড়া", "हावड़ा"),
    ("PS_0502", "DIST_05", "Shibpur PS", "Sibpur", "শিবপুর", "शिवपुर"),
    ("PS_0503", "DIST_05", "Bally PS", "Bali", "বালি", "बाली"),
    ("PS_0504", "DIST_05", "Uluberia PS", "Uluberiya", "উলুবেড়িয়া", "उलूबेड़िया"),
    ("PS_0505", "DIST_05", "Golabari PS", "Gola Bari", "গোলাবাড়ি", "गोलाबाड़ी"),
    ("PS_0601", "DIST_06", "Chinsurah PS", "Chuchura|Chinsura|Chunchura", "চুঁচুড়া|চুচুড়া", "चुंचुड़ा|चिनसुरा"),
    ("PS_0602", "DIST_06", "Chandannagar PS", "Chandan Nagar|Chandernagore", "চন্দননগর", "चंदननगर"),
    ("PS_0603", "DIST_06", "Serampore PS", "Srirampur|Serampur", "শ্রীরামপুর|সেরামপুর", "श्रीरामपुर|सेरामपुर"),
    ("PS_0604", "DIST_06", "Rishra PS", "Risra", "রিষড়া", "रिषड़ा"),
    ("PS_0605", "DIST_06", "Tarakeswar PS", "Tarakeshwar", "তারকেশ্বর", "तारकेश्वर"),
    ("PS_0701", "DIST_07", "Berhampore PS", "Baharampur|Berhampur|Baharampore", "বহরমপুর", "बहरमपुर"),
    ("PS_0702", "DIST_07", "Murshidabad PS", "", "মুর্শিদাবাদ", "मुर्शिदाबाद"),
    ("PS_0703", "DIST_07", "Jiaganj PS", "Jiyaganj", "জিয়াগঞ্জ", "जियागंज"),
    ("PS_0704", "DIST_07", "Jangipur PS", "Jongipur", "জঙ্গিপুর", "जंगीपुर"),
    ("PS_0705", "DIST_07", "Kandi PS", "Kandy", "কান্দি", "कांदी"),
    ("PS_0706", "DIST_07", "Domkal PS", "Domkol", "ডোমকল", "डोमकल"),
]

# code, district code, name, unit type
SUBDIVISIONS_BLOCKS = [
    ("SB_0101", "DIST_01", "English Bazar", "Block / Municipality"),
    ("SB_0102", "DIST_01", "Chanchal", "Subdivision / Block"),
    ("SB_0201", "DIST_02", "Raiganj", "Subdivision / Block"),
    ("SB_0202", "DIST_02", "Islampur", "Subdivision / Block"),
    ("SB_0301", "DIST_03", "Balurghat", "Sadar Subdiv / Municipality"),
    ("SB_0302", "DIST_03", "Gangarampur", "Subdivision / Block"),
    ("SB_0401", "DIST_04", "Kolkata Municipal Corp", "Municipal Corporation"),
    ("SB_0501", "DIST_05", "Howrah Sadar", "Subdivision / Corporation"),
    ("SB_0502", "DIST_05", "Uluberia", "Subdivision"),
    ("SB_0601", "DIST_06", "Chinsurah", "Sadar Subdivision"),
    ("SB_0602", "DIST_06", "Chandannagar", "Subdivision / Corporation"),
    ("SB_0603", "DIST_06", "Srirampur (Serampore)", "Subdivision"),
    ("SB_0701", "DIST_07", "Berhampore", "Sadar Subdivision"),
    ("SB_0702", "DIST_07", "Lalbagh", "Subdivision"),
    ("SB_0703", "DIST_07", "Jangipur", "Subdivision"),
]

# code, district code, police station code (None = the PDF gives none that exists), name, place type, aliases_en, aliases_bn, aliases_hi
PLACES = [
    ("PL_0101", "DIST_01", "PS_0101", "Rathbari", "Locality / Transit Hub", "", "রথবাড়ি", "रथबाड़ी"),
    ("PL_0102", "DIST_01", "PS_0101", "Mangalbari", "Commercial Area", "Mongolbari", "মঙ্গলবাড়ি", "मंगलबाड़ी"),
    ("PL_0103", "DIST_01", "PS_0102", "Gour", "Historical Site", "Gaur|Gaud|Gor", "গৌড়|গৌর", "गौड़"),
    ("PL_0104", "DIST_01", "PS_0103", "Pandua", "Historical Site", "Panduya", "পান্ডুয়া|পাণ্ডুয়া", "पांडुआ"),
    ("PL_0105", "DIST_01", "PS_0104", "Sujapur", "Commercial / Village", "", "সুজাপুর", "सुजापुर"),
    ("PL_0201", "DIST_02", "PS_0201", "Karnajora", "Administrative Hub", "Karnojora", "করণজোড়া", "करणजोड़ा"),
    ("PL_0202", "DIST_02", "PS_0202", "Aluabari", "Railway Junction", "Aluyabari", "আলুয়াবাড়ি", "आलुआबाड़ी"),
    ("PL_0203", "DIST_02", "PS_0203", "Dalkhola More", "Highway Crossing", "Dalkola More", "ডালখোলা মোড়", "डालखोला मोड़"),
    ("PL_0301", "DIST_03", "PS_0301", "Mangalpur", "Urban Locality", "Mongolpur", "মঙ্গলপুর", "मंगलपुर"),
    ("PL_0302", "DIST_03", "PS_0303", "Buniadpur", "Municipal Town", "Buniyadpur", "বুনিয়াদপুর", "बुनियादपुर"),
    ("PL_0401", "DIST_04", "PS_0401", "Camac Street", "Commercial District", "Kamak Street", "ক্যামাক স্ট্রিট", "कैमक स्ट्रीट"),
    ("PL_0402", "DIST_04", "PS_0402", "Esplanade", "Transit Hub / Square", "Esplanade|Espleneid", "এসপ্ল্যানেড", "एस्प्लेनेड"),
    ("PL_0403", "DIST_04", "PS_0403", "Hastings", "Neighborhood", "", "হেস্টিংস", "हेस्टिंग्स"),
    (
        "PL_0404",
        "DIST_04",
        "PS_0404",
        "Shyambazar",
        "Commercial Node",
        "Shyam Bazar|Syambazar|Shamba zar",
        "শ্যামবাজার",
        "श्यामबाज़ार|श्यामबाजार",
    ),
    (
        "PL_0405",
        "DIST_04",
        "PS_0406",
        "Ruby Crossing",
        "Major Junction",
        "Ruby More|Rubi Crossing",
        "রুবি ক্রসিং|রুবি মোড়",
        "रूबी क्रॉसिंग",
    ),
    ("PL_0406", "DIST_04", None, "Gariahat", "Shopping Hub", "Gariyahat|Gariahaat", "গড়িয়াহাট", "गरियाहाट|गड़ियाहाट"),
    ("PL_0501", "DIST_05", "PS_0501", "Howrah Maidan", "Civic Center", "Howrah Moidan", "হাওড়া ময়দান", "हावड़ा मैदान"),
    ("PL_0502", "DIST_05", "PS_0502", "Nabanna", "Administrative HQ", "Nobanno", "নবান্ন", "नबन्ना"),
    ("PL_0503", "DIST_05", "PS_0503", "Liluah", "Industrial Hub", "Lilua", "লিলুয়া", "लिलुआ"),
    ("PL_0504", "DIST_05", "PS_0504", "Bauria", "Town Area", "Baouria|Bauriya", "বাউড়িয়া", "बाउरिया"),
    ("PL_0601", "DIST_06", "PS_0601", "Bandel", "Railway Hub", "Bondel", "ব্যান্ডেল", "बंडेल"),
    (
        "PL_0602",
        "DIST_06",
        "PS_0602",
        "Strand Road (Chandannagar)",
        "Heritage Promenade",
        "Strand Road",
        "স্ট্র্যান্ড রোড",
        "स्ट्रैंड रोड",
    ),
    ("PL_0603", "DIST_06", "PS_0603", "Uttarpara", "Municipal Area", "Uttarpada", "উত্তরপাড়া", "उत्तरपाड़ा"),
    (
        "PL_0604",
        "DIST_06",
        "PS_0605",
        "Tarakeswar Precinct",
        "Religious Site",
        "Tarakeswar Mandir",
        "তারকেশ্বর মন্দির",
        "तारकेश्वर मंदिर",
    ),
    (
        "PL_0701",
        "DIST_07",
        "PS_0701",
        "Cossimbazar",
        "Historical Town",
        "Kasimbazar|Cossim Bazar",
        "কাশিমবাজার",
        "कासिमबाज़ार",
    ),
    (
        "PL_0702",
        "DIST_07",
        "PS_0702",
        "Hazarduari Precinct",
        "Heritage Area",
        "Hazarduari|Hazar Duari",
        "হাজারদুয়ারি",
        "हज़ारद्वारी",
    ),
    ("PL_0703", "DIST_07", "PS_0703", "Azimganj", "Historical Town", "Ajimganj|Azim Ganj", "আজিমগঞ্জ", "अज़ीमगंज"),
    ("PL_0704", "DIST_07", "PS_0704", "Farakka", "Infrastructure Node", "Faraka|Farrakka", "ফারাক্কা", "फरक्का"),
]

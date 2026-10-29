from agent import where_doctor_sits as w


def test_floor_and_room_questions_are_recognised():
    for t in ["ডাক্তার আজ কোন তলায় বসবেন", "डॉक्टर आज किस मंज़िल पर बैठेंगे", "which floor will the doctor sit today", "ডাক্তার কোথায় বসেন", "which room number is the doctor in"]:
        assert w.asks_floor(t), t


def test_other_sentences_are_not():
    for t in ["ডাক্তার আজ কখন বসবেন", "ডাঃ সুতপা চ্যাটার্জী কি আছেন", "will the doctor sit today", "তলা"]:
        assert not w.asks_floor(t), t

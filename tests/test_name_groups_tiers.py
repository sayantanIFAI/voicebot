"""Owner, 2026-10-30: "ডক্টর অভিজিৎ সেন" (Dr. Abhijit Sarkar and Dr. Biswajit Sen exist, nobody is Abhijit Sen) is answered "nobody of that name sits here", then the doctor with the SAME FIRST NAME --
never "Abhijit Sarkar or Biswajit Sen?", where Biswajit Sen shares only the surname."""

from agent import name_groups

DOCS = [
    {"name": "Dr. A. Sarkar", "full_name": "Dr. Abhijit Sarkar", "full_name_bn": "ডাঃ অভিজিৎ সরকার", "full_name_hi": "डॉ. अभिजीत सरकार"},
    {"name": "Dr. B. Sen", "full_name": "Dr. Biswajit Sen", "full_name_bn": "ডাঃ বিশ্বজিৎ সেন", "full_name_hi": "डॉ. बिस्वजीत सेन"},
    {"name": "Dr. Arindam Sen", "full_name": "Dr. Arindam Sen", "full_name_bn": "ডাঃ অরিন্দম সেন", "full_name_hi": "डॉ. अरिंदम सेन"},
]


def test_a_first_name_that_exists_hides_the_surname_only_doctors():
    g = name_groups.build("ডক্টর অভিজিৎ সেন", DOCS, ["Dr. Abhijit Sarkar", "Dr. Biswajit Sen"])
    assert g is not None
    assert [i["full_name"] for i in g["given"]] == ["Dr. Abhijit Sarkar"]
    assert g["surname"] == []
    said = name_groups.reply(g, "ডক্টর অভিজিৎ সেন", "bn")
    assert said.startswith("'অভিজিৎ সেন' নামে কেউ বসেন না।")
    assert "অভিজিৎ সরকার" in said and "বিশ্বজিৎ" not in said and "অরিন্দম" not in said


def test_with_no_first_name_match_the_surname_doctors_are_offered():
    g = name_groups.build("সুব্রত সেন", DOCS, [])
    assert g is not None and {i["full_name"] for i in g["surname"]} == {"Dr. Biswajit Sen", "Dr. Arindam Sen"} and g["given"] == []

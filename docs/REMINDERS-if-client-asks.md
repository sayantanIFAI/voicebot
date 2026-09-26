# Reminders: two things to build ONLY IF THE CLIENT ASKS

Nothing here is in the code. This file is only a reminder so the two ideas are not forgotten. If the client asks for either
one, start from the notes below.

---

## 1. Prescription OCR over WhatsApp

**The idea.** A caller sends a photo of a prescription on WhatsApp. The bot reads it (OCR), and answers correctly from what it
says: for example which tests were prescribed, so it can quote prices, preparation and offer to book them.

**To build if asked:**
- A WhatsApp inbound channel that receives the image (and links it to the call or the caller's phone number).
- An OCR step for handwritten and printed prescriptions, in English and Bengali or Hindi where the doctor wrote them.
- Match what was read to the clinic's own test list. A name that does not match confidently is asked back to the caller,
  never guessed (the same rule as "Doctor Nobody").

**Rules that must hold (from CLAUDE.md):**
- OCR output is a *reading of an image*, not a fact. The bot may say what it read ("I can see CBC and Lipid Profile on the
  prescription, is that right?") and must get a yes before acting on it. It never interprets medicines, doses or diagnoses.
- Prices, preparation and timings still come from the clinic API, never from the model.
- A blurred or unreadable photo gets a polite "please send a clearer photo" or a hand-off to staff.

**Also decide first:** consent and retention (how long the image is kept, who can see it; India's DPDP Act), where it is stored,
and whether a staff member reviews unclear ones.

---

## 2. Payment link on WhatsApp, and checking that the booking is paid

**The idea.** After a booking, the bot sends a payment link on WhatsApp. Later it can tell whether the booking has been paid.

**To build if asked:**
- A payment provider (for example UPI or a card gateway) that creates a payment link for a booking's amount.
- Send the link on WhatsApp to the number on the booking.
- The paid / not paid status comes from the provider's confirmation (a webhook), **not from the caller saying "I paid"**.
  The bot may only say a booking is paid when the provider has said so.
- A payment status on the booking record, and a reply when the caller asks "did my payment go through?".

**Things to plan for:**
- Retries and duplicates: use the idempotent-write support already in the clinic API (`Idempotency-Key`) so a repeated
  request never creates a second payment link or a second booking.
- What "unpaid" means for the booking: hold the slot for how long, when it is released, and how that fits with the existing
  cancellation charge rules.
- Refunds and failed payments, and what the bot says in each case.
- No card, UPI PIN or bank details are ever spoken or typed into the call. The link takes the caller to the provider's page.

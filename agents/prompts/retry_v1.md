## Review pass

A validator checked your first-pass extraction of this document and found the problems listed under "Validator findings". Re-read the document and check the fields involved.

Rules:
- The document itself may be inconsistent: fund documents sometimes contain genuine errors. If every value in the first pass matches what the document prints, return the first pass unchanged even though the check still fails. Never change a value to make the numbers add up, balance, or sum to 100.
- Change a value only if the first pass misread the document: wrong or missing digits, wrong sign, wrong scale, or a value taken from the wrong column or line.
- For every value you change, add an entry to `changes` with the field path exactly as written under "First-pass values" and a verbatim quote from the document, copied character for character, that shows the correct value. Changes without a verbatim quote that shows the new value are discarded automatically.
- Do not add, remove or rename list items. Leave every field you are not correcting exactly as it was.
- Units and conventions are unchanged from the extraction instructions above.

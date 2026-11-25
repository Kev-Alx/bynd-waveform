# Validation of 76 remaining historical utterances

The [manifest](/workspace/improvement/execution_refine/validation-76.jsonl) contains the 76 remaining utterances of the historical English validation100, after removing the 24 added to development48. Original order, audio paths, reference text and metadata are preserved. SHA256: `3747ca5b726ef899c24bc8f92e83ba44f086a537bffcf98b9bcd52cba71df30a`. Selection uses no recognition, attack or quality outcomes.

All 76 selected audio hashes and native 16 kHz mono metadata match their original manifests. Whole utterances span 2.010–21.930 seconds, totaling 524.905 seconds across 34 speakers. Nothing was resampled, cropped or normalized.

There is no ID, compressed-audio hash or normalized-transcript overlap with development48, confirmation48, final200 or training. Final200 and training also share no speakers. Shared dev-clean speakers remain: 26 with development48 and 26 with confirmation48. This is utterance-disjoint validation, with historical exposure and shared speakers; it cannot support fresh-test or speaker-independent claims. Protected manifests were read only as metadata; protected audio and outcomes were not opened.

Freeze the candidate method, code, hyperparameters and development48 gate decision before validating. Evaluate all candidate and baseline conditions on these same 76 inputs, references and fixed native source/MOS policy. Any further method changes after seeing these outcomes make this pool development data. Future pilots keep the user's requested 48 utterances. Confirmation48 and final200 remain untouched.

The [receipt](/workspace/improvement/execution_refine/validation-76.receipt.json) binds the script, all source manifests and every selected audio file. Rerunning the script verifies the frozen artifacts and refuses a changed selection.

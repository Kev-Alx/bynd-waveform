"""Both cohort sizes must remain complete and paired before source inference."""
import pytest
import runtime_refine
from source_evaluate_refine import validate_pilot_records


def fixture(count):
    allowed = {str(i): {'text': f'reference {i}'} for i in range(count)}
    records = [{'id': uid, 'text': row['text'], 'audio': {v: {} for v in runtime_refine.VARIANTS}}
               for uid, row in allowed.items()]
    return records, allowed


@pytest.mark.parametrize('count', [48, 76])
def test_complete_registered_cohort_is_accepted(count):
    records, allowed = fixture(count)
    validate_pilot_records(records, allowed, count)


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'reference', 'variant', 'foreign', 'size'])
def test_bad_cohort_is_rejected_before_model_inference(failure):
    records, allowed = fixture(76)
    count = 76
    if failure == 'missing': records.pop()
    elif failure == 'duplicate': records[-1] = records[0]
    elif failure == 'reference': records[-1]['text'] = 'changed'
    elif failure == 'variant': records[-1]['audio'].pop('constrained100')
    elif failure == 'foreign': records[-1]['id'] = 'sealed-reserve'
    else: count = 48
    with pytest.raises(ValueError): validate_pilot_records(records, allowed, count)

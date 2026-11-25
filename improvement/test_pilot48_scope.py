"""The actual evaluator must reject partial pilots and sealed/changed inputs."""
from copy import deepcopy
import pytest
import runtime48
from source_evaluate48 import validate_pilot_records


def fixture_records():
    allowed = {str(i): {'text': f'original reference {i}'} for i in range(48)}
    records = [{'id': uid, 'text': row['text'],
                'audio': {v: {} for v in runtime48.VARIANTS}}
               for uid, row in allowed.items()]
    return records, allowed


def test_all48_paired_variants_are_accepted():
    records, allowed = fixture_records()
    validate_pilot_records(records, allowed)


@pytest.mark.parametrize('failure', ['47', 'duplicate', 'foreign', 'reference', 'missing_variant'])
def test_partial_or_unregistered_pilot_is_rejected_before_model_inference(failure):
    records, allowed = fixture_records()
    if failure == '47':
        records.pop()
    elif failure == 'duplicate':
        records[-1] = deepcopy(records[0])
    elif failure == 'foreign':
        records[-1]['id'] = 'sealed-confirmation-or-final'
    elif failure == 'reference':
        records[-1]['text'] = 'changed reference'
    else:
        records[-1]['audio'].pop('pgd_full')
    with pytest.raises(ValueError):
        validate_pilot_records(records, allowed)

"""The reader separates checked CPU examples from pending native qualification."""
import json

from mastery_hub.read import study_edition_notice


def test_only_registered_study_sections_receive_a_notice(tmp_path):
    assert study_edition_notice('5.6',tmp_path) is None
    path=tmp_path/'mastery_hub/study_editions.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'schema_version':1,'sections':{'5.6':{'reference_state':'pending'}}}))
    assert study_edition_notice('7.1',tmp_path) is None
    assert 'check pending' in study_edition_notice('5.6',tmp_path)


def test_cpu_execution_does_not_promote_native_or_learner_acceptance(tmp_path):
    path=tmp_path/'mastery_hub/study_editions.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'schema_version':1,'sections':{'5.6':{'reference_state':'cpu_reference_checked'}}}))
    text=study_edition_notice('5.6',tmp_path)
    assert 'CPU worked reference executed' in text
    assert 'pretrained/native validation and learner acceptance remain pending' in text
    assert 'local downloads' in text

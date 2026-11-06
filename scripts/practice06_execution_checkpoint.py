"""Finite actual 0.6 publication candidate; no runtime, count or output creation.

The exact observed execution slots and explicit native text projections are pinned
by source. Root publication still requires independent bound review and adoption.
Only private constants are substituted by distinct tiny DEV guard tests.
"""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.rome_runtime_metadata_checkpoints import _json, _read, _record, _typed

_ADOPTION = {'contract_path': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/contract.json',
 'record': {'sha256': 'b5869694a40a54fbdcd0bf09719c7bff8373b839a928d9b780616db75250f52f',
            'size_bytes': 82659}}
_TARGET = "chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_solutions.ipynb"
_IDENTITY = {"section": "0.6", "chapter": "chapter0_fundamentals", "exercise_dir": "part6_fake_interpretability_results_practice", "page": "06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice"}
_MEMBERS = {"before": "before.json", "after": "after.json", "observations": "observations.json", "receipt": "receipt.json"}

_SCOPE = 'actual_0.6_independently_adopted_execution'

_SCHEMAS = {'contract': 'arena.practice06.execution-publication.contract.v1',
 'sidecar': 'arena.practice06.execution-publication.sidecar.v1',
 'observations': 'arena.practice06.execution-publication.observations.v1',
 'receipt': 'arena.practice06.execution-publication.receipt.v1'}

_SHAPE = (107, 36)

_BEFORE_RECORD = {'sha256': '99a54f5da0316fee8ddca826bd76be1eff7997e58c7b8b97676f9a4b528bc1e7', 'size_bytes': 118866}

_SIGNATURES = [(4,
  'foundation-b2d8aa7d821ab9b1dafbcee5368d92e2',
  'ba9d07cb8c89198a5f61d8699cc6285fb65b94c25db885430e67ac53e21935ed'),
 (6,
  'foundation-9e5a51d4a174e004bf38d6bfa731f117',
  '6258b24969fe1f22c03bb95f62c5c0a85644e9162c996a1ece5594b475efbf73'),
 (7,
  'foundation-31bce0d7e03653376f0ba6145e3f3218',
  'e0d59e213c3fc7342febf684b2198a62645a1b4238497b2de0f4c8e328ce335b'),
 (8,
  'foundation-f874d4abdad1b98b6e2039e9ba90f45c',
  '316765aff4b47c8909587972b30f93dd6a68432ee96afeec36ae0832db20691b'),
 (9,
  'foundation-0509b834b12ea05ae9d6f75317d01e87',
  'e7a57365cf50bac2767ff51bd090858909090784e73d9f0ef7255dcaa59815a9'),
 (10,
  'foundation-ff37bcebae1865a0c21e06f49b5eb87b',
  'a19a33717e0512b5fff4591f41170f80bfbf16d2f634283d62d0fd2257a05881'),
 (11,
  'foundation-b671777f3c0be0417bcbb406c7cd56b7',
  '79612d46f68cfa5d8ed4446e58e514d061639198bdcae94c26c809f54c52650c'),
 (14,
  'foundation-8d0e1e1327e9708ee398e0df5184a328',
  '0726b6ec3090545a6cae64100d3b70b35bfcb0e7e551e828efc7bb243d7c787c'),
 (15,
  'foundation-cca57dea81905f82fa137728438ae17e',
  '5e396bc7d70567da4b3181f36a9097922d9094389c7d690549cd2a2efc5e94c5'),
 (16,
  'foundation-871a114ed0b46b4ea10055eb45f27c32',
  '5ceb7b9aad98a4d621f590a9c5d8019c71521aa985ca7e00d0da7767d4fd9cb3'),
 (17,
  'foundation-3916a5f8c031b416c747bb460ef014a6',
  '519b5f33534f5a80a61c0d792ad0c0ba0761d1f2a1796f9a1f9d230b0884068a'),
 (21,
  'foundation-4fa8bda4b8c2f8c84b3091eb2d0db334',
  '2d43d2fd20545f2183b148e506d1498c377ff692f8af9b299600cc8a2f1378bc'),
 (22,
  'foundation-6dcc765c7264794fdcfc2e1ca8621db4',
  '429558145faf8302f9d377ea68cae94c53a90a9528b135fc551ea80b8d5d1a34'),
 (23,
  'foundation-8e7206889dd4db785f84936da1478bb9',
  '50d8de9b826bb2e9cb40c0e351935b680cfb4f84e4e28d0d58377446296103d3'),
 (27,
  'foundation-c625b04c1ef733d2bc96cba769169e29',
  '311cfb6f8e8da72e08830455d08456d2846236d90c97f232dd347faefd4a74c6'),
 (28,
  'foundation-4999f331b6a5d2eef0533c733bf85325',
  '20a3af3277b9a673f41a65bd62d15a2e4bf34404cea79886a373f5fc995e98a4'),
 (29,
  'foundation-d1ae0b0e514d976836e4a940ca6a1731',
  'df8e6b027877dce49138667e76ef1a83359c371a2dfd9fd264f0036f941e9f0b'),
 (30,
  'foundation-256539b8ebd709b74b164b06ea355c3a',
  '3f17e0309cc6fe5c7b366b848a479690f8ce9d00df90344a02fe2c28f0b29888'),
 (34,
  'foundation-3db2d38af8fdf5f0809b987bdbe19417',
  '59bfc226c8dd8bc382a4124a6726e9386e6430733abd0d3841b328a8f9b7a200'),
 (35,
  'foundation-0b7927152dc6dd71ff01044d0479ea72',
  '5ee233049d409e7974b32125c719b3ee3f422a1422cc61586f2832fba8e5214c'),
 (36,
  'foundation-2de462bbdb09fc6cf2a66e70b5b2f28e',
  'f4b867393f850a5286849d145cbca74f76db5c2806dce0df132e452665f9da72'),
 (40,
  'foundation-f391080371d2a8320b681ef1a34ec742',
  'c28159dda482bd369b2911b94e0c3ae636f7f3f816e2f296f34c474ef52e86a7'),
 (41,
  'foundation-747f6947af28ef61c61441ee13d2812e',
  'ea46c051f8ffc2b93530f1f2a65607c436cce48130c56f5806c1dd601d62a1fc'),
 (42,
  'foundation-fdf1c4b871189cf3953093b799558837',
  '1d14cccdcd8cd6d312772182ba64a1f49677b4a766eb813a4dcefc580384f594'),
 (43,
  'foundation-09a33df008620f9b96f272e66897de65',
  'bfbab0b044a2970375ae22c5482b4ce4a2a7b94a53778f63c5ef65774fb7310f'),
 (44,
  'foundation-39b9fd88b8372411f6180a0f8df6a16a',
  '7013e0211f762a15450dc046a5f7b4d082dae594624523ee755db6adc65f7907'),
 (45,
  'foundation-2d5868114553f1046d6316164bc60b56',
  'd1de13d6a11561b5d58cfeb8cef8a5534faea261151b1c26999ef5f65224c469'),
 (46,
  'foundation-be88060aac3143311626f2d4ea16d384',
  'a05164fa2ebe521fdd25d4e25ee185d20a5704b1637915c0550578bba0438eae'),
 (47,
  'foundation-6aed5fce5dff847d2a86f96b8fdde8f8',
  '4ea68447f4d765211a68f1513c51f20236bf24b39085968567fb7db767b0247c'),
 (53,
  'foundation-9d327f228d67cb6b51ef88e0672b343a',
  '4c9be3735f70aea304d9c4cfac4b970cf66b1da15d35c86388cb4607b11f53f6'),
 (56,
  'foundation-186e38ec4dbd96f1e5dada33228317e8',
  '84e045c5e8f29f4a6b55d5c8ae0cfe0327f323562605e65320b1963390533146'),
 (57,
  'foundation-a41bc049d61e441ed41b1dfd6536614c',
  'a6e1420c43032370d340f8024a5cf36588608d29fb8e842805ab16bb628d7bb7'),
 (99, 'cf15837c', '8994b083c072d6b48a2ff089bbe8f727ff1dacf6d544f3b6aafbb44b79fbac76'),
 (101, 'adf01886', 'b951a2048bb3f521521e74b89dfc6d53a8a2d08c2f7f5956bfc08121de7635f2'),
 (103, 'ed65a25b', 'ebc91df21d50adfd315a4d73af707ef821d358009ed4f5446c6da3af60a61705'),
 (105, '7bae0967', '8772f2a6fe21cfcf547be1bb770579840541fc63254c8739db4e18c0363d39cc')]

_DEPENDENCIES = {'.venv/lib/python3.14/site-packages/nbformat/__init__.py': {'sha256': 'e4afe87976b229354971db0bb3c69b7eb775b2c7f0f13486ed910489f59f1509',
                                                             'size_bytes': 6230},
 '.venv/lib/python3.14/site-packages/nbformat/_version.py': {'sha256': '54ada94fcf3cbb3ed176e79fbd3ef373ceb0848248210afdd89ada4bfcc3de59',
                                                             'size_bytes': 816},
 '.venv/lib/python3.14/site-packages/nbformat/notebooknode.py': {'sha256': 'c2028aec0519f09f04aeaaa89c0664d45da6e71f0de64b7fe9b87ba0116e891a',
                                                                 'size_bytes': 3193},
 '.venv/lib/python3.14/site-packages/nbformat/reader.py': {'sha256': '9c1b9311d55ac9bb44c390dd8e8ee2eb29e2768ff03c4e08d105b5a83f5f4d4d',
                                                           'size_bytes': 2704},
 '.venv/lib/python3.14/site-packages/nbformat/v4/__init__.py': {'sha256': '4889a60715ab854544ee4d89d1bff6166ef742fd5eb504918914813b1bda2587',
                                                                'size_bytes': 819},
 '.venv/lib/python3.14/site-packages/nbformat/v4/nbjson.py': {'sha256': '06437722e594a5354813324f2cbb8bb5517ebba980fe190eba8107b192ac10e5',
                                                              'size_bytes': 2030},
 '.venv/lib/python3.14/site-packages/nbformat/v4/rwbase.py': {'sha256': '1ea4961fb5986965642d4d122f71c7c0e44b2e7db5eaa9eb872019ae28fe60a4',
                                                              'size_bytes': 4289},
 '.venv/lib/python3.14/site-packages/nbformat/validator.py': {'sha256': '3db1fe48de1060981b13bcf828b8a5779ae30dd082ec06f712f066e90fbbfd46',
                                                              'size_bytes': 22349},
 '.venv/lib/python3.14/site-packages/torch/__init__.py': {'sha256': 'd9dfff4b75d46e4c75572200a3466b70231d05b0318e38ac1bd121789165fb49',
                                                          'size_bytes': 109312},
 '.venv/lib/python3.14/site-packages/torch/version.py': {'sha256': '38fbf6a65805efde37fb1773c0ce4eeceeb29cefb5895e6b1690109d0bbbe6a6',
                                                         'size_bytes': 317},
 '.venv/pyvenv.cfg': {'sha256': 'd933e3b78120f8d4ac9acc041ff95d9c31e9a405d06a0d18992ffd627eb6d4f5',
                      'size_bytes': 202},
 'arena_ext/__init__.py': {'sha256': '98abfe2501c14bb586c6e5711446dd613a44fde9c7e0fc1b6d1438a986e58b4e',
                           'size_bytes': 2917},
 'arena_ext/fake_interpretability.py': {'sha256': '7af3fb39f6dbf36a1f593ebb6e8649ce4284191433721ee20c33eb81def6cf30',
                                        'size_bytes': 9935},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results/artifacts.lock.yml': {'sha256': 'cb2a8244545cc21e8fb1b86a927f46c84b72efacd79e59b5ab6230f67259c8cc',
                                                                                            'size_bytes': 2573},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results/expected_outputs/README.md': {'sha256': '99c760babe697fb5b668ff8ed2e301fe6afb6c0b8d70e7c9ebb3475fbab68ea6',
                                                                                                    'size_bytes': 1803},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results/solutions.py': {'sha256': 'ed3b167b3061d6017117fd64b0eb4cb304a6f56011b4d849f1507b10c2f5aafa',
                                                                                      'size_bytes': 13608},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results/tests.py': {'sha256': '3203678914ad059aac35e838308100fa87a72005887e6841e84e63ddaa8e7679',
                                                                                  'size_bytes': 13249},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results/verification_report.json': {'sha256': '0d787413e969200e277d16a1eec2023fd731f8b152e5ef882d0553298d127f8b',
                                                                                                  'size_bytes': 9960},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_exercises.ipynb': {'sha256': 'e13c2a0ecb12df834e7193b4759187cd34c855228e7fd37582e2a5e585f2b415',
                                                                                                                                                                   'size_bytes': 131030},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb': {'sha256': '1c24972df726d0c808cfb0c654b150ef4de4d5f6b86b415145cf35d774ca5d00',
                                                                                                                                                          'size_bytes': 116215},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/README.md': {'sha256': 'a2e078ac0a75492b9524e8a36841b027acf505083a65cc79b683fc45c33bdbbb',
                                                                                            'size_bytes': 2914},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/artifacts.lock.yml': {'sha256': 'cb2a8244545cc21e8fb1b86a927f46c84b72efacd79e59b5ab6230f67259c8cc',
                                                                                                     'size_bytes': 2573},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/expected_outputs/fake_interpretability_signature_result.svg': {'sha256': '994cad5444f744cb8ef05ab14bb17075bf2273f12e1c15a60d3d6a6c01f29d90',
                                                                                                                                              'size_bytes': 2936},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/media/fake_interpretability_signature_result.svg': {'sha256': '994cad5444f744cb8ef05ab14bb17075bf2273f12e1c15a60d3d6a6c01f29d90',
                                                                                                                                   'size_bytes': 2936},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/solutions.py': {'sha256': 'd2f393afbd1ba5e1aa62f0d16641410fe9edba93e745d2dbdcb4dc088831d56e',
                                                                                               'size_bytes': 15812},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/tests.py': {'sha256': '3203678914ad059aac35e838308100fa87a72005887e6841e84e63ddaa8e7679',
                                                                                           'size_bytes': 13249},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/utils.py': {'sha256': '31724e90fc6aa0a8d0345c2ca84358b4fe0afd6bf6b93e9f3aaa0a7ef46dba28',
                                                                                           'size_bytes': 141},
 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/verification_report.json': {'sha256': '0d787413e969200e277d16a1eec2023fd731f8b152e5ef882d0553298d127f8b',
                                                                                                           'size_bytes': 9960},
 'docs/mastery/evidence/integrated_root_cpu_after_reviewed_additions60_20261021/pytest.txt': {'sha256': 'd7087e98abf6ccfcbd0785c37236f388aa18d1f4fc35395419fee0bfdfe80bca',
                                                                                              'size_bytes': 27739},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before.tar.gz': {'sha256': '03a9f7c06b19982a776b474bd86aee12967421700cf30712c4aff29f7d5e42e0',
                                                                          'size_bytes': 4724306},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before.tar.gz.review-bundle.json': {'sha256': 'ecc91b2885712c48f1d7545809a77bacbc7707958dff15a916df2c399554d876',
                                                                                             'size_bytes': 3997},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before_manifest.json': {'sha256': '1b15fca79d95a98f0cb2efb94350f4e7badba748bf64590999248751c07a1c54',
                                                                                 'size_bytes': 2380},
 'docs/mastery/evidence/lesson_integration_20261021_q008/navigation_inverse.json': {'sha256': '426ce9753c3084e44839262c27032f420a3ee4adbdf40c831d4a054bc83a56a8',
                                                                                    'size_bytes': 18533},
 'docs/mastery/evidence/lesson_integration_20261021_q008/original_replay_path_proof.json': {'sha256': '74b90697db610e2f8b602a957f927eb961184acb02553c96f5007fec853c603f',
                                                                                            'size_bytes': 107065},
 'docs/mastery/evidence/lesson_integration_20261021_q008/original_source_preservation_baseline.json': {'sha256': '339d29a7a502d693efebbc5900da3f6b55babc559e311019816704b9ebc8c927',
                                                                                                       'size_bytes': 81446},
 'docs/mastery/evidence/lesson_integration_20261021_q008/practice_book_projection_inverse.json': {'sha256': 'ccc486d0682c9f039c734c4eacc0d12ba4a90b665aaa6abeac06acb030d2013d',
                                                                                                  'size_bytes': 5923482},
 'docs/mastery/evidence/lesson_integration_20261021_q008/practice_mapping.json': {'sha256': '9c3644d795d3bcdcfeee1802fcb6f779b7092882a21302a1403e1d9850fcdd4c',
                                                                                  'size_bytes': 2047},
 'docs/mastery/evidence/lesson_integration_20261021_q008/raw.tar.gz': {'sha256': 'e07f8759aca35be63571f8bcbd9a622612a853650e7ce6f726747d24d5f8046b',
                                                                       'size_bytes': 527039},
 'docs/mastery/evidence/lesson_integration_20261021_q008/raw.tar.gz.review-bundle.json': {'sha256': '4fd454cbf14e334aff55910225468019bcbed11d6516f97e8c787420b99f7ba3',
                                                                                          'size_bytes': 3986},
 'docs/mastery/evidence/lesson_integration_20261021_q008/raw_manifest.json': {'sha256': 'f92b794ec1b85bb73ecf0ff8351301034ddf0f7e1102dd78fc7732c2c43aac8e',
                                                                              'size_bytes': 2376},
 'infrastructure/chapters/chapter0_fundamentals/master_0_6.ipynb': {'sha256': '86bca5712becc46b218daeef6e7aaf279429452e21a584c7c0c0fca86fa551e2',
                                                                    'size_bytes': 124625},
 'infrastructure/chapters/chapter0_fundamentals/master_0_6.py': {'sha256': 'f392724397dd092a015738071865bb30f17ab0b304d8b915f97541a097f8faae',
                                                                 'size_bytes': 93754},
 'infrastructure/extension/source_preservation.py': {'sha256': '0b28c52b73a3ede9c669e561b6dff0a3bf4d594ff45dee67ec96b59a36a511dc',
                                                     'size_bytes': 9235},
 'infrastructure/extension/source_preservation_contracts.json': {'sha256': '1f05a959af920e750b739989357451719a90241f1220ace9f3f5fe72bfc58c4b',
                                                                 'size_bytes': 1066337},
 'pyproject.toml': {'sha256': '77a8312a7b1ed75962fa0e2be53ec8a03e5d2c60279f42c5ff9291c6ac8d0aac',
                    'size_bytes': 3066},
 'requirements.txt': {'sha256': '87de9e75adc76a55d2abc2773fc87ed6cde9ad2b2b7b0fe83a21d566a8392966',
                      'size_bytes': 2733},
 'scripts/evidence_bundles.py': {'sha256': 'c166452c18f6d5aa8b9de81371b24a1e99cc296d080d9308f738cb4b1d9547f8',
                                 'size_bytes': 18258},
 'scripts/rome_runtime_metadata_checkpoints.py': {'sha256': '941f8fa50bab2870b1eb573a5990163ebe364349aab002629775dffd3f4f9acf',
                                                  'size_bytes': 15186},
 'tests/test_extension_sections.py': {'sha256': '7ca6da2ddb17a38e2d78782b9dc31617cf1c76dc512284f479a27186fdcdb1b2',
                                      'size_bytes': 21349},
 'tests/test_q008_practice_projection.py': {'sha256': 'd1c841418f9c3ba8bb94f6dc9c807cb59902bb805ed67a9006471fef6aa96a54',
                                            'size_bytes': 7229},
 'uv.lock': {'sha256': 'e96e946bb00f50592d32c1b3bf453fca4482a0097731120defbebcd978f019eb',
             'size_bytes': 610963}}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _decode(raw):
    value = _json(raw)
    _typed(value)  # Also rejects finite-looking JSON tokens that overflow to inf.
    return value


def _same(left, right):
    return _typed(left) == _typed(right)


def _bundle(root, descriptor):
    expected_keys = {'archive_path', 'archive', 'sidecar_path', 'sidecar', 'manifest_path', 'manifest', 'files'}
    _require(type(descriptor) is dict and set(descriptor) == expected_keys, 'Unknown finite bundle descriptor')
    captured = {}
    for role in ('archive', 'sidecar', 'manifest'):
        captured[role] = _read(root, descriptor[role + '_path'])
        _require(_same(_record(captured[role]), descriptor[role]), 'Bundle ' + role + ' identity changed')
    manifest, sidecar = _decode(captured['manifest']), _decode(captured['sidecar'])
    _require(set(manifest) == {'scope', 'files'} and manifest['scope'] == _SCOPE, 'Unknown finite manifest')
    _require(set(sidecar) == {'schema', 'scope', 'archive', 'manifest', 'files'} and sidecar['schema'] == _SCHEMAS['sidecar'] and sidecar['scope'] == _SCOPE, 'Unknown finite sidecar')
    _require(_same(manifest['files'], descriptor['files']) and _same(sidecar['files'], descriptor['files']), 'Bundle member map changed')
    _require(_same(sidecar['archive'], descriptor['archive']) and _same(sidecar['manifest'], descriptor['manifest']), 'Sidecar provenance changed')
    _require(set(descriptor['files']) == set(_MEMBERS.values()), 'Unknown or missing finite bundle members')
    _require(len(captured['archive']) <= 64 * 1024 * 1024, 'Finite bundle exceeds source boundary')
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(captured['archive']), mode='r:*') as archive:
        members = archive.getmembers()
        _require(len(members) == len(_MEMBERS) and {m.name for m in members} == set(_MEMBERS.values()), 'Unknown, duplicate or missing archive members')
        for member in members:
            _require(member.isfile() and not member.issparse() and member.size <= 16 * 1024 * 1024, 'Unsafe archive member')
            raw = archive.extractfile(member).read()
            _require(_same(_record(raw), descriptor['files'][member.name]), 'Archive member bytes changed')
            payloads[member.name] = raw
    return {role: payloads[name] for role, name in _MEMBERS.items()}


def _message(message, kind, request):
    _require(type(message) is dict and message.get('msg_type') == kind, 'Wrong raw message kind')
    _require(type(message.get('parent_header')) is dict and message['parent_header'].get('msg_id') == request, 'Wrong raw message parent')
    _require(type(message.get('content')) is dict, 'Missing raw message content')
    return message['content']


def _validate_objects(payloads, contract):
    before, after, observations, receipt = (_decode(payloads[role]) for role in ('before', 'after', 'observations', 'receipt'))
    _require(_same(_record(payloads['before']), _BEFORE_RECORD) and _same(_BEFORE_RECORD, contract['before']), 'Frozen authored before object changed')
    _require(_same(_record(payloads['after']), contract['after']), 'Adopted complete after bytes changed')
    _require(type(before.get('cells')) is list and type(after.get('cells')) is list and len(before['cells']) == len(after['cells']) == _SHAPE[0], 'Whole cell count changed')
    code_indices = [i for i, cell in enumerate(before['cells']) if cell.get('cell_type') == 'code']
    _require(len(code_indices) == _SHAPE[1] and code_indices == [row[0] for row in _SIGNATURES], 'Authored code cohort changed')
    _require(len(contract['code_entries']) == len(code_indices), 'Missing adopted whole code cells')
    _require(set(observations) == {'schema', 'scope', 'attempt_id', 'rows'} and observations['schema'] == _SCHEMAS['observations'] and observations['scope'] == _SCOPE, 'Unknown raw observation shape')
    _require(set(receipt) == {'schema', 'scope', 'attempt_id', 'status', 'claim', 'cleanup', 'ordered_requests'} and receipt['schema'] == _SCHEMAS['receipt'] and receipt['scope'] == _SCOPE, 'Unknown or partial receipt')
    attempt = contract['attempt_id']
    _require(type(attempt) is str and bool(attempt) and observations['attempt_id'] == receipt['attempt_id'] == attempt and receipt['status'] == 'completed', 'Wrong or unsuccessful attempt')
    claim, cleanup = receipt['claim'], receipt['cleanup']
    _require(set(claim) == {'attempt_id', 'claim_id', 'exclusive', 'consumed'} and claim['attempt_id'] == attempt and type(claim['claim_id']) is str and bool(claim['claim_id']) and claim['exclusive'] is True and claim['consumed'] is True, 'Missing exclusive consumed claim')
    _require(set(cleanup) == {'kernel_id', 'status', 'normal'} and type(cleanup['kernel_id']) is str and bool(cleanup['kernel_id']) and cleanup['status'] == 'reaped' and cleanup['normal'] is True, 'Missing normal cleanup/reaping')
    rows = observations['rows']
    _require(type(rows) is list and len(rows) == len(code_indices), 'Partial raw observation cohort')
    inverse, forward = copy.deepcopy(after), copy.deepcopy(before)
    requests, counts = [], []
    for signature, entry, row in zip(_SIGNATURES, contract['code_entries'], rows, strict=True):
        index, cell_id, source_pin = signature
        _require(set(entry) == {'index', 'before_cell', 'after_cell'} and type(entry['index']) is int and entry['index'] == index, 'Adopted code position changed')
        old, new = before['cells'][index], after['cells'][index]
        _require(old['id'] == cell_id and hashlib.sha256(_typed(old['source']).encode()).hexdigest() == source_pin, 'Authored cell identity/source changed')
        _require(_same(old, entry['before_cell']) and _same(new, entry['after_cell']), 'Complete adopted code object changed')
        count = new.get('execution_count')
        _require(type(count) is int and count > 0 and type(new.get('outputs')) is list, 'Missing exact actual count/output objects')
        # Copy only provided actual execution values for comparison, never synthesize
        # a count, output, timing value, or generic metadata exemption.
        permitted = copy.deepcopy(old)
        permitted['execution_count'] = count
        permitted['outputs'] = copy.deepcopy(new['outputs'])
        _require(_same(permitted, new), 'Unadopted source/ID/type/metadata/field change')
        _require(all(type(output) is dict and output.get('output_type') in {'stream', 'display_data', 'execute_result'} for output in new['outputs']), 'Error or unknown output object')
        _require(set(row) == {'index', 'id', 'source', 'request_id', 'kernel_id', 'after_cell', 'raw_after_cell', 'projection', 'iopub', 'shell_reply'}, 'Unknown raw row field')
        _require(type(row['index']) is int and row['index'] == index and row['id'] == cell_id and _same(row['source'], old['source']) and _same(row['after_cell'], new), 'Raw source-correlated whole cell differs')
        _check_projection(row, old, new)
        request = row['request_id']
        _require(type(request) is str and bool(request) and request not in requests and row['kernel_id'] == cleanup['kernel_id'], 'Raw request/kernel identity changed')
        requests.append(request)
        counts.append(count)
        reply = _message(row['shell_reply'], 'execute_reply', request)
        _require(reply.get('status') == 'ok' and type(reply.get('execution_count')) is int and reply['execution_count'] == count, 'Shell reply/count differs')
        events = row['iopub']
        _require(type(events) is list and len(events) >= 3, 'Partial raw IOPub transport')
        _require(_message(events[0], 'status', request).get('execution_state') == 'busy' and _message(events[-1], 'status', request).get('execution_state') == 'idle', 'Incomplete raw execution state')
        input_events = [event for event in events[1:-1] if event.get('msg_type') == 'execute_input']
        _require(len(input_events) == 1 and events[1] is input_events[0], 'Missing, duplicate or misordered raw execute_input')
        inputs = _message(input_events[0], 'execute_input', request)
        source = old['source'] if type(old['source']) is str else ''.join(old['source'])
        _require(inputs.get('code') == source and type(inputs.get('execution_count')) is int and inputs['execution_count'] == count, 'Raw input source/count differs')
        output_events = events[2:-1]
        raw_outputs = row['raw_after_cell']['outputs']
        _require(len(output_events) == len(raw_outputs), 'Raw output count differs')
        for output, event in zip(raw_outputs, output_events, strict=True):
            content = _message(event, output['output_type'], request)
            _require(_same({key: value for key, value in output.items() if key != 'output_type'}, content), 'Raw output bytes/types differ; explicit serialization review required')
        inverse['cells'][index] = copy.deepcopy(old)
        forward['cells'][index] = copy.deepcopy(new)
    _require(counts == sorted(set(counts)) and receipt['ordered_requests'] == requests, 'Actual counts/request order changed')
    _require(_same(inverse, before) and _same(forward, after), 'Full typed inverse/forward changed an unrelated object')
    return before, after


def _guard(root):
    if _ADOPTION is None:
        raise RuntimeError('0.6 publication binding pending independently accepted actual execution and review')
    _require(type(_ADOPTION) is dict and set(_ADOPTION) == {'contract_path', 'record'}, 'Unknown source adoption binding')
    raw = _read(root, _ADOPTION['contract_path'])
    _require(_same(_record(raw), _ADOPTION['record']), 'Source-pinned publication contract changed')
    contract = _decode(raw)
    _require(set(contract) == {'schema', 'scope', 'target', 'identity', 'before', 'after', 'bundle', 'attempt_id', 'source_pins', 'code_entries'}, 'Unknown publication contract field')
    _require(contract['schema'] == _SCHEMAS['contract'] and contract['scope'] == _SCOPE and contract['target'] == _TARGET and _same(contract['identity'], _IDENTITY), 'Wrong finite publication identity')
    _require(_same(contract['source_pins'], _DEPENDENCIES), 'Source/dependency cohort changed')
    for relative, expected in _DEPENDENCIES.items():
        _require(_same(_record(_read(root, relative)), expected), 'Pinned source/dependency changed: ' + relative)
    payloads = _bundle(root, contract['bundle'])
    before, after = _validate_objects(payloads, contract)
    physical = _read(root, _TARGET)
    _require(physical == payloads['after'] and _same(_record(physical), contract['after']), 'Physical file differs from exact adopted after state')
    checked = {'before_bytes': payloads['before'], 'after_bytes': physical, 'before': before, 'after': after}
    _validate_actual_provenance(root, checked, _decode(payloads['observations']), _decode(payloads['receipt']))
    return checked


def preflight_practice06_publication(*, root, identity, master_path):
    named = Path(master_path).name == 'master_0_6.py'
    if identity.get('section') != _IDENTITY['section'] and not named:
        return None
    _require(_same(identity, _IDENTITY) and named, 'Wrong registered 0.6 build identity')
    return _guard(root)


def restore_practice06_notebooks(**kwargs):
    # Resolve at call time to retain existing preservation monkeypatch boundaries.
    from infrastructure.extension.source_preservation import restore_notebooks
    if kwargs['identity'].get('section') != _IDENTITY['section']:
        return restore_notebooks(**kwargs)
    # The registered source-pinned recipe cannot be replaced by a caller override.
    _require(kwargs.get('contracts_path') is None, 'Registered 0.6 forbids contracts_path overrides; pinned canonical recipe required')
    checked = preflight_practice06_publication(root=kwargs['root'], identity=kwargs['identity'], master_path=kwargs['master_path'])
    existing = kwargs.get('existing_notebooks')
    _require(type(existing) is dict and set(existing) == {'exercises', 'solutions'}, 'Missing complete existing notebook set')
    supplied = existing['solutions']
    if supplied is None:
        # Existing source-parity tests convert a validated copied master into an
        # empty temporary chapter. The canonical physical after is still required.
        canonical_master = Path(kwargs['root']) / 'infrastructure/chapters/chapter0_fundamentals/master_0_6.py'
        _require(Path(kwargs['master_path']).resolve() != canonical_master.resolve(), 'Missing existing canonical solutions payload')
    else:
        supplied = supplied.encode() if type(supplied) is str else supplied
        _require(type(supplied) is bytes and supplied == checked['after_bytes'], 'Supplied existing book differs from checked physical after bytes')
    arguments = dict(kwargs)
    arguments['existing_notebooks'] = dict(existing, solutions=checked['before_bytes'].decode('utf-8'))
    restored = restore_notebooks(**arguments)
    _require(type(restored) is dict and set(restored) == {'exercises', 'solutions'} and _same(_decode(restored['solutions']), checked['before']), 'Old recipe did not return exact historical restored object')
    _guard(kwargs['root'])  # A callback cannot conceal intervening file/dependency drift.
    return dict(restored, solutions=checked['after_bytes'].decode('utf-8'))


def recover_practice06_checkpoint_bytes(root, relative_path, actual_bytes):
    _require(type(relative_path) is str and relative_path == _TARGET and type(actual_bytes) is bytes, 'Unknown checkpoint path or input type')
    checked = _guard(root)
    _require(actual_bytes == checked['after_bytes'], 'Supplied bytes differ from checked physical after bytes')
    return checked['before_bytes']


def load_practice06_checkpoint_json(root, path):
    path = Path(path)
    if path.absolute() == Path(root) / 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb':
        # One reviewed adapter only: validate captured bytes before any import/use.
        adapter_path = 'scripts/practice16_execution_checkpoint.py'
        adapter_bytes = _read(root, adapter_path)
        _require(_same(_record(adapter_bytes), {'sha256': 'c57b8f717c4f2a4694ea4c89ae6c2c929b6c202a2b8ade2e434e85078fd9d8c7', 'size_bytes': 96986}), 'Pinned 1.6 adapter source changed')
        namespace = {'__name__': 'scripts._verified_practice16_checkpoint', '__file__': str(Path(root) / adapter_path)}
        exec(compile(adapter_bytes, str(Path(root) / adapter_path), 'exec'), namespace)
        actual = _read(root, 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb')
        return _decode(namespace['recover_practice16_checkpoint_bytes'](root, 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb', actual))
    if path.absolute() != Path(root) / _TARGET:
        return json.loads(path.read_text())
    actual = _read(root, _TARGET)
    return _decode(recover_practice06_checkpoint_bytes(root, _TARGET, actual))

_PROVENANCE = {'docs/mastery/evidence/practice06_execution_publication_20261023_v2/bundle.tar.gz.review-bundle.json': {'sha256': 'b8392e6b7e784753f83f5dc088f6d30d0d6a72d0b4ed19084c45df77927f6963',
                                                                                                         'size_bytes': 1890},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/actual_controller_exit.json': {'sha256': 'ef8960769fa25c2b77a63322154ad22476e580bafb5333bae9b8f187ef0e903e',
                                                                                                                                    'size_bytes': 816},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/actual_stdout_stderr.log': {'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                                                                                                                                 'size_bytes': 0},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/fresh_source_and_authority_check.json': {'sha256': 'b0f1827c232a4b430161cbf17accb43d0ad451a9edf37afc1dfe9930d0879728',
                                                                                                                                              'size_bytes': 1098},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/root_preexecution_adoption.json': {'sha256': 'ac9f451263c912fd9dcbdc198a8c775dabe6b9f734bd0375d0888a004326f726',
                                                                                                                                        'size_bytes': 1587},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/FROZEN.json': {'sha256': '7a412b6fbd653813442762a8c8070b23b341314e43053c67ffa9893c8fc3b980',
                                                                                                                          'size_bytes': 905},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/attempt_claim.json': {'sha256': '97101b89f88adf1029ef1803560ddfc6c036716ac2b61fa4fb1b48d2cb18c594',
                                                                                                                                 'size_bytes': 754},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/packet_manifest.json': {'sha256': '0b8ef9747173d007c61ad5b780d7b1a1cdfbe313b6ffdcb8bbf2b138ad23f391',
                                                                                                                                   'size_bytes': 25359},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/authority.json': {'sha256': '01fd75216dba67a924a06c22d46d7345d592dcf9df6377554a5318f77b9bf962',
                                                                                                                                      'size_bytes': 561},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/before_notebook.ipynb': {'sha256': '99a54f5da0316fee8ddca826bd76be1eff7997e58c7b8b97676f9a4b528bc1e7',
                                                                                                                                             'size_bytes': 118866},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_004.after_at_reply.json': {'sha256': 'd0062762b18294c42a00486bcca3cb83618d43b5e43c264235ddc37aaf7ac090',
                                                                                                                                                    'size_bytes': 313},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_004.attempt.json': {'sha256': 'c50ab1b52d28eb8bf8bbcc295b2ea4353d63f111a6b4cfd4f1da8f26234ecbe0',
                                                                                                                                             'size_bytes': 4718},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_006.after_at_reply.json': {'sha256': '532cffb3e7056c7942253146c0a90d6d4d4bd66bdcac7ff0995af396757f3c0c',
                                                                                                                                                    'size_bytes': 1237},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_006.attempt.json': {'sha256': '8b07008dbc2b0c0b3a2e35bffc33327bd1b04f79fef624bf08b1ea55e123ffab',
                                                                                                                                             'size_bytes': 5252},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_007.after_at_reply.json': {'sha256': 'a099500b113caffb0d35198478badcd01cdac98ffd802bb2f0fcef9daaf05f35',
                                                                                                                                                    'size_bytes': 337},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_007.attempt.json': {'sha256': '75202d4b79031b3ef8246d713f9bc8dcffd29f2c353cefd32c391e31e0f41226',
                                                                                                                                             'size_bytes': 3452},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_008.after_at_reply.json': {'sha256': '3451bb2b79e89af97a08d75b38fd43fd28f0a51de1f7f3aa104da1cb54544147',
                                                                                                                                                    'size_bytes': 346},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_008.attempt.json': {'sha256': '1a743ee9f7dd295141cdf81e4242865236ef3b0154fd1d9afe11b23c69fd04f7',
                                                                                                                                             'size_bytes': 3470},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_009.after_at_reply.json': {'sha256': '80f85e92ab0106d8eba0218c19fa8d09b204f8485b1d5029c105d15c5158cc48',
                                                                                                                                                    'size_bytes': 295},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_009.attempt.json': {'sha256': '55389099ad516a73bfb821306184892be07836756e5da301b7de433dab18ec21',
                                                                                                                                             'size_bytes': 3368},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_010.after_at_reply.json': {'sha256': '3234d681fbe01ef1cae46eff23ab203beafdae61315045fbfdb334c56119fbf8',
                                                                                                                                                    'size_bytes': 352},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_010.attempt.json': {'sha256': '4be70bd3758757d8621270510c139dea26a578e108781bc4e81dfcd1d586cd1a',
                                                                                                                                             'size_bytes': 3482},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_011.after_at_reply.json': {'sha256': 'e2bfde6957a075e3e9fea4865371864fcc1745d6638272a7a1fdd58cd013b15d',
                                                                                                                                                    'size_bytes': 360},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_011.attempt.json': {'sha256': '326a2c4900fe14793d3c08de0306baa87b3275f43f798e671df5e279df0d2b65',
                                                                                                                                             'size_bytes': 3498},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_014.after_at_reply.json': {'sha256': 'da9354fb3a32ad8f791ff77f69852889fe1097c4e06fc309486bc803efcd16a9',
                                                                                                                                                    'size_bytes': 523},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_014.attempt.json': {'sha256': '467e22602362b0c8c4b480c4e11435b5381aae767f5377eb11ff137d5b8eefa5',
                                                                                                                                             'size_bytes': 3824},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_015.after_at_reply.json': {'sha256': '966a986ac28cd54139c60212bd1e1a797dd478559abf3433cefdea9441240391',
                                                                                                                                                    'size_bytes': 427},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_015.attempt.json': {'sha256': '9a68ca0729df7e05937307928f4926642afe032582b2eeb5e1efd23830489329',
                                                                                                                                             'size_bytes': 3636},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_016.after_at_reply.json': {'sha256': 'c5d6249447369cf74925df9a8d156dd1b29be44c4ac0cd0b33459fb1b3868b85',
                                                                                                                                                    'size_bytes': 1337},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_016.attempt.json': {'sha256': 'b182008dc4f1db44182f7362723b29b93167171c96b8c47c24d021bb956f7e39',
                                                                                                                                             'size_bytes': 5457},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_017.after_at_reply.json': {'sha256': '32822bec824aca54573bed55d64049ac4ab7ba06f782d2d2f61e881dd8d67ade',
                                                                                                                                                    'size_bytes': 644},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_017.attempt.json': {'sha256': 'a0e333bf739633e67c27342609855812f4f0777ac5d6353388350f58333edb71',
                                                                                                                                             'size_bytes': 4681},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_021.after_at_reply.json': {'sha256': 'b5768aad2dec47a873b8554fb3ed70f271c32ea5ecd1c12366a95bcdcdef4a57',
                                                                                                                                                    'size_bytes': 672},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_021.attempt.json': {'sha256': '5193d269371687dccca54ae065938ec3309b5e4881dbdeace0c4d938645dcd64',
                                                                                                                                             'size_bytes': 4127},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_022.after_at_reply.json': {'sha256': 'f03ce48fb508c1926b372dcbac6a10ab5d8b43eb812a690fd0264f2f52d91da2',
                                                                                                                                                    'size_bytes': 1311},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_022.attempt.json': {'sha256': '0135a2160d97d467a4e62aa68c2420936b84281b671186afa251f47095feca17',
                                                                                                                                             'size_bytes': 5405},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_023.after_at_reply.json': {'sha256': 'dc0531a8689fcabb1aba433b88c2a9a2575592f029a32d7e7f87770260242400',
                                                                                                                                                    'size_bytes': 508},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_023.attempt.json': {'sha256': '4ccedced549b5392e3ab4cb10a7ca3e798ece530ed3f5163d4d80d000548b492',
                                                                                                                                             'size_bytes': 4409},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_027.after_at_reply.json': {'sha256': '12e2d021e94a798acff7f583425f3011a12f0907fa135556f710c60f203b5036',
                                                                                                                                                    'size_bytes': 702},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_027.attempt.json': {'sha256': 'ffcee8cd470873bb1ecd6224b4f3abd57b9b54fb11d9da4178c2f12cc7fdfc53',
                                                                                                                                             'size_bytes': 4187},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_028.after_at_reply.json': {'sha256': '7a980fa698c2153b0828b0345df4b8ec6336b075999ae5d172aff285d72c7663',
                                                                                                                                                    'size_bytes': 492},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_028.attempt.json': {'sha256': 'b67d8920aa4e010ed23f39b698fc0aeea54c25fdc5612cc8861ba8d3f7faec7c',
                                                                                                                                             'size_bytes': 3767},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_029.after_at_reply.json': {'sha256': '4be010782c23b1713209db5c43ce637f3a8d92b32a7efc525759fa7197966eff',
                                                                                                                                                    'size_bytes': 1422},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_029.attempt.json': {'sha256': '8520eddf8b885f5e37c16388afcc804aa766d034e8e986b147ff66f90f0f96a7',
                                                                                                                                             'size_bytes': 5627},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_030.after_at_reply.json': {'sha256': '928a611d79e7e85ff0b4d71c04a6074c2552b2e931e462b7fa2260620def8365',
                                                                                                                                                    'size_bytes': 486},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_030.attempt.json': {'sha256': '8372814b29efc27395443dda28cd604b778014a978806b2413f54d4c582732b1',
                                                                                                                                             'size_bytes': 4365},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_034.after_at_reply.json': {'sha256': '39f9d3b713368014116b482e8166ea4d5a94d96b7f2eda246d7406b1a256facf',
                                                                                                                                                    'size_bytes': 856},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_034.attempt.json': {'sha256': '18a88f1365e2ecf8dafc4a7948a56bca9d6acc41d0dfc6dbdfcaaebef0d31275',
                                                                                                                                             'size_bytes': 4495},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_035.after_at_reply.json': {'sha256': 'af057a02c7bf50f020689fb1669a39fab4e5173056c62ea964241030b0140c00',
                                                                                                                                                    'size_bytes': 1544},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_035.attempt.json': {'sha256': 'bd1ccb0d411ef5dac7044bda52bbdef937b3a7d10fe20a5c9cc89433c94233c5',
                                                                                                                                             'size_bytes': 5871},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_036.after_at_reply.json': {'sha256': '419a478c2cc73d15a139a2dbde7dfce8d6405928a9b5256957c0dac1d93f722e',
                                                                                                                                                    'size_bytes': 536},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_036.attempt.json': {'sha256': 'fbdea13f2780e6d044545500f54ed4167bde6a75cf8e99500a7645fb3205bceb',
                                                                                                                                             'size_bytes': 4465},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_040.after_at_reply.json': {'sha256': '7b813a632d05b40b1f43e757448cba66dcd600a3a9b51f3b7373a6e658a9fee8',
                                                                                                                                                    'size_bytes': 1169},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_040.attempt.json': {'sha256': '9aac19fe76ac6f60511c1b7ec86f62e5a003cfb567ea211d3759bb1411537ea9',
                                                                                                                                             'size_bytes': 5129},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_041.after_at_reply.json': {'sha256': 'e97c43d0cde5679a8d2887774ef9e0bbc5f31865f7e4b57c5791c1bc7a422586',
                                                                                                                                                    'size_bytes': 209},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_041.attempt.json': {'sha256': '0c42a0c5b582f0b7faf30684aea252c79db40bc7779250bda321227bd59c7c9f',
                                                                                                                                             'size_bytes': 3209},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_042.after_at_reply.json': {'sha256': '49be3d4d13e046891186bae41ca201353acc7d99b0849cfc845759bcd2bdbe3d',
                                                                                                                                                    'size_bytes': 211},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_042.attempt.json': {'sha256': '48c2fa965c7a62fb78ee2a8e7856b35c32d48726912d81b009be2fc891e199a3',
                                                                                                                                             'size_bytes': 3213},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_043.after_at_reply.json': {'sha256': 'fcae125f8d07f926ed2da457c8e263afe47431f67ba814770e86426fe8f68a2f',
                                                                                                                                                    'size_bytes': 215},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_043.attempt.json': {'sha256': '00d9574927b27bc745f0476bc7b406b3ce6b06e1bf332cd6ee8731442817ab82',
                                                                                                                                             'size_bytes': 3221},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_044.after_at_reply.json': {'sha256': '0cacdb84cfebfc9e4f64a61fa1324d78fc345beb09335a5606d9a6fcc51a66c9',
                                                                                                                                                    'size_bytes': 229},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_044.attempt.json': {'sha256': 'e773c282ac957510bd22b3355576991d3511f94fb567aaa5ac0e21e0f24afe71',
                                                                                                                                             'size_bytes': 3249},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_045.after_at_reply.json': {'sha256': '7050d23e7a60379da207eb3502185b5cc03236722b6a2967a2619f9648028d60',
                                                                                                                                                    'size_bytes': 211},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_045.attempt.json': {'sha256': 'b33c034497ce08cffcbd6b9a09057b60d2823e75ffe252dba68af75ccdae1c30',
                                                                                                                                             'size_bytes': 3213},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_046.after_at_reply.json': {'sha256': 'd7d3b67c9e89c441908494bc53a979e4224b061c6d1bebe3792f1e534445e3ac',
                                                                                                                                                    'size_bytes': 552},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_046.attempt.json': {'sha256': '94e4b020df9891848b7ea8abc002a6f71228716b23914745fa0532da3569f902',
                                                                                                                                             'size_bytes': 3895},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_047.after_at_reply.json': {'sha256': 'db673a0b0bafec8a754b7fae1a749992f7888a6810e2f2da8c73f610cc5eb1fc',
                                                                                                                                                    'size_bytes': 448},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_047.attempt.json': {'sha256': '240a050e0f2663499f1569c1d1e8661af34624b991933aac68461bf8887fabd7',
                                                                                                                                             'size_bytes': 4299},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_053.after_at_reply.json': {'sha256': '59736b51de714238727c423e3db9435b711dac834d8bd4e41d624414a5a7732c',
                                                                                                                                                    'size_bytes': 464},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_053.attempt.json': {'sha256': '054c0a534a99a45809127e89057a549eb8539f482ea2a37a48b45cdace3cae5c',
                                                                                                                                             'size_bytes': 4331},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_056.after_at_reply.json': {'sha256': 'c221cdc75c6f8cbedeb8127bc1b59ff510d18eda0c276bb774140c32710d2f74',
                                                                                                                                                    'size_bytes': 2580},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_056.attempt.json': {'sha256': '640e051c3163535b040874ba05cb2c5e1aec484f9c6350fdf2b32ba5d778bbd7',
                                                                                                                                             'size_bytes': 7951},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_057.after_at_reply.json': {'sha256': 'a8beadbeb1539e09873e1498e419b241a800e371ac48b3b1ee788200ee4fab98',
                                                                                                                                                    'size_bytes': 327},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_057.attempt.json': {'sha256': 'd472c3e7bd7e622d5df0d750393470ae222f5b18b8c8c346132f07547e1501d8',
                                                                                                                                             'size_bytes': 3445},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_099.after_at_reply.json': {'sha256': '3c531b82433409d3c72cda508f7eedd76e571b087b64f8ed37f1d75e26f16194',
                                                                                                                                                    'size_bytes': 941},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_099.attempt.json': {'sha256': '4f0419b8d3ce9cd6435ef2d1d027ca7fc159bd01b49845a191404d92600d8b35',
                                                                                                                                             'size_bytes': 4491},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_101.after_at_reply.json': {'sha256': 'a5e369695aff1cf94f30f3b4462793da8ed90828b5ec93b73494cd5f28339ba6',
                                                                                                                                                    'size_bytes': 2196},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_101.attempt.json': {'sha256': '023b7c78f14957e1e5c68ef5af71cc5143db027b8e229110db351e786cb887a1',
                                                                                                                                             'size_bytes': 7613},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_103.after_at_reply.json': {'sha256': '18744a2f388f3398cdb5accb03ff4f18cca2f067b0d0deec56da7cd219c2f009',
                                                                                                                                                    'size_bytes': 1876},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_103.attempt.json': {'sha256': '46f33eabecb1e0c8f23c6cbd971bae5269b1be272fcf36f4777de76b84f4f2c2',
                                                                                                                                             'size_bytes': 6981},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_105.after_at_reply.json': {'sha256': 'fa2354cf04e80123fdce88888ad8c95c999819aab139967f5ebc61e4e8b39119',
                                                                                                                                                    'size_bytes': 1923},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cell_105.attempt.json': {'sha256': '79479fdd2726a0af355adabc0ed566e1f73c2b3e83b3d1105ccaabc400c4e56b',
                                                                                                                                             'size_bytes': 7686},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cleanup_owned_identity.json': {'sha256': '2f38f8321b469e3ea76e5cb34735b1296aab0c66cb72b0817df5f24bb32e3d39',
                                                                                                                                                   'size_bytes': 760},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/completion.json': {'sha256': 'f2850c88ee6647fb6507ffc61efc4ccd13725db22793f1b1aea80c604de694dc',
                                                                                                                                       'size_bytes': 420},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/environment.json': {'sha256': 'ebeea15ae07f921c0cbefbf8723465c64417de6962c02ffb8ea181980d744e0d',
                                                                                                                                        'size_bytes': 2526},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/kernel_stderr.txt': {'sha256': '3f2b766f08dfd18a853f26878cf7e0d8cad05ab72687a30d2e374da957f6a5a5',
                                                                                                                                         'size_bytes': 283},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/kernel_stdout.txt': {'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                                                                                                                                         'size_bytes': 0},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/namespace_observation.json': {'sha256': '6d5ed37942739a3204a5fdf80ebbf052d932772e917707e228cd3579a68b0c56',
                                                                                                                                                  'size_bytes': 1363},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/owned_kernel_identity.json': {'sha256': '2f38f8321b469e3ea76e5cb34735b1296aab0c66cb72b0817df5f24bb32e3d39',
                                                                                                                                                  'size_bytes': 760},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/partial_notebook.ipynb': {'sha256': 'd03f7015d6e53153b9b62856de299fcf2767f9d05dc62e0d717b32c2b33c43dd',
                                                                                                                                              'size_bytes': 103770},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/progress.json': {'sha256': 'ec8fc4fd97a198e51abc3bd3d7811a96845dd6ac72e0d4947cc2daf455897ed5',
                                                                                                                                     'size_bytes': 14771},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/raw_transport.jsonl': {'sha256': 'b8730efa507cb39f139872b55bd3e73f0aa26212f67fa982658f790e9504ef84',
                                                                                                                                           'size_bytes': 188798},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/result.json': {'sha256': '841627619f1f73d363a735a0357c8390673578a0a3a0955035494b3b4f5591c0',
                                                                                                                                   'size_bytes': 15846},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/state/ipythondir/profile_default/startup/README': {'sha256': '638ec6b584245b2ea977392aca550d2fa7814993142111b04f15cf505fc10417',
                                                                                                                                                                       'size_bytes': 371},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/terminal_pending.json': {'sha256': '2b1abc68b9de8dfc19fa8fd63d018dd111bb52ebeb815d8745455998d4f9aa5a',
                                                                                                                                             'size_bytes': 492},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/whole_before_code_objects.json': {'sha256': '725e46ec583301cc953301cffcc5178d874ef1a87d5f63f3cc7120472ae2f31f',
                                                                                                                                                      'size_bytes': 29283},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_review_v3/FROZEN.json': {'sha256': 'cb557c75a1fece923e950ef89ac0ab5d0092e6b461a60ad308fc9b831a117bf8',
                                                                                                                     'size_bytes': 712},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_review_v3/review.json': {'sha256': '78a21fe3e4d14469010c359ea34437042b2f60e33c9fd086c38eacab32884e34',
                                                                                                                     'size_bytes': 1105},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/FROZEN.json': {'sha256': '35d69619fd9d4af77213c8169cbb47dd2794babeca12d0436d74074f224c1e05',
                                                                                           'size_bytes': 815},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/REVIEW.md': {'sha256': 'fc33ec455374aedca180be19f84f33da0e6bd2efb86959d78d7ee18581b676d6',
                                                                                         'size_bytes': 7295},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/actual_source_manifest.json': {'sha256': '4f9c04e71527c487267bfbb3f6bfabbb27c376b17fe4688b96a75fb6721afa15',
                                                                                                           'size_bytes': 26503},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/cell_observations.json': {'sha256': 'a221b198ff38bb82cd1f505abe723545a48d3722417f9b99e37668a70c226823',
                                                                                                      'size_bytes': 20204},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/prior_consumed_ledger_check.json': {'sha256': '775442b6e5ab00b187ccb141216e208df9cf8bd2b186e34cb04a1819d3814ff6',
                                                                                                                'size_bytes': 2006},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/raw_export_map.json': {'sha256': '1e95f419cf74097fa82b1ee510eaa796e2e7c458a144b9af62252f50d5c62407',
                                                                                                   'size_bytes': 46430},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/review.json': {'sha256': 'f2082a6f85bda24d54cdc732b6a948b676286632595dd769cdef8c464019ea00',
                                                                                           'size_bytes': 3618},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/review_manifest.json': {'sha256': 'e1f1d4e191a2ab464b4856f565e8e0f534f164b6d97233d718d9b86f4f15780a',
                                                                                                    'size_bytes': 1064},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/verify_actual.log': {'sha256': '492325494d2465e0d0eb98dadba9252be71b29a1bf1e0e29eaad17faf621d516',
                                                                                                 'size_bytes': 2282},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/verify_actual.py': {'sha256': '55e363251be9abf1ff56cd214ac9607d664a5fdfcf344125700420867d75ee3e',
                                                                                                'size_bytes': 17763},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/runtime_input_manifest.json': {'sha256': 'a4a255eba45b95b6f0bff986df252c7908d3f4790a54fcb4b36905b5b79695e7',
                                                                                                    'size_bytes': 58215},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/source_copy_map.json': {'sha256': '9e57d0e8f9b0bed384db7bc9536809b58b26ebcb71b13c10e825037b97913e05',
                                                                                             'size_bytes': 45514},
 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/typed_projection_witness.json': {'sha256': 'a88c83f7f3ca43bf3849f4be01368ed0d9a47bba9c4bf1df1e366045e363e2d8',
                                                                                                      'size_bytes': 69665}}

_ACTUAL_ROLES = {'actual_export': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/raw_export_map.json',
 'actual_manifest': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/actual_source_manifest.json',
 'actual_review': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/review/review.json',
 'authority.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/authority.json',
 'before_notebook.ipynb': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/before_notebook.ipynb',
 'cleanup_owned_identity.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/cleanup_owned_identity.json',
 'completion.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/completion.json',
 'consumed_claim': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/attempt_claim.json',
 'controller_exit': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/actual_controller_exit.json',
 'input_manifest': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/runtime_input_manifest.json',
 'namespace_observation.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/namespace_observation.json',
 'owned_kernel_identity.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/owned_kernel_identity.json',
 'partial_notebook.ipynb': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/partial_notebook.ipynb',
 'projection_witness': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/typed_projection_witness.json',
 'raw_transport.jsonl': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/raw_transport.jsonl',
 'result.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/result.json',
 'runtime_adoption': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_dispatch/root_preexecution_adoption.json',
 'whole_before_code_objects.json': 'docs/mastery/evidence/practice06_execution_publication_20261023_v2/raw/practice06_runtime_preparation_v3/run_once/whole_before_code_objects.json'}


def _check_projection(row, old, new):
    """Validate precisely witnessed source/text representation changes, no repair."""
    raw, witness = row['raw_after_cell'], row['projection']
    _require(type(witness) is dict and set(witness) == {'rule', 'canonical_source', 'raw_source', 'output_fields'} and witness['rule'] == 'exact_observed_source_join_and_text_split_v1', 'Unknown typed projection rule')
    _require(type(old['source']) is list and all(type(line) is str for line in old['source']) and type(raw.get('source')) is str, 'Source projection types changed')
    _require(_same(witness['canonical_source'], old['source']) and witness['raw_source'] == raw['source'] == ''.join(old['source']), 'Raw joined source differs from exact canonical source array')
    expected_raw = copy.deepcopy(old)
    expected_raw['source'] = raw['source']
    expected_raw['execution_count'] = new['execution_count']
    expected_raw['outputs'] = copy.deepcopy(raw.get('outputs'))
    _require(_same(raw, expected_raw), 'Raw nonexecution field or actual count changed')
    projected = copy.deepcopy(raw['outputs'])
    fields, used = witness['output_fields'], set()
    _require(type(fields) is list, 'Missing finite output field witnesses')
    for field in fields:
        _require(type(field) is dict and set(field) == {'output_index', 'path', 'raw', 'published'}, 'Unknown output field witness')
        index, path = field['output_index'], field['path']
        _require(type(index) is int and 0 <= index < len(projected) and type(path) is list and path in (['text'], ['data', 'text/plain']), 'Unknown non-text projection path')
        key = (index, tuple(path))
        _require(key not in used, 'Duplicate output field witness')
        used.add(key)
        output = projected[index]
        _require((path == ['text'] and output.get('output_type') == 'stream') or (path == ['data', 'text/plain'] and output.get('output_type') == 'execute_result'), 'Unobserved MIME/output projection')
        container = output
        for component in path[:-1]:
            _require(type(container.get(component)) is dict, 'Missing exact output container')
            container = container[component]
        value = container.get(path[-1])
        split = field['published']
        _require(type(value) is str and value == field['raw'] and type(split) is list and all(type(line) is str for line in split) and value.splitlines(True) == split and ''.join(split) == value, 'Text split/join or trailing newline differs')
        container[path[-1]] = copy.deepcopy(split)
    _require(_same(projected, new['outputs']), 'Unwitnessed or non-text output change')


def _validate_actual_provenance(root, checked, observations, receipt):
    # Empty provenance exists only in clearly separate private DEV test fixtures.
    if not _PROVENANCE:
        return
    captured = {}
    for relative, expected in _PROVENANCE.items():
        raw = _read(root, relative)
        _require(_same(_record(raw), expected), 'Accepted actual evidence changed: ' + relative)
        captured[relative] = raw
    def raw(role):
        return captured[_ACTUAL_ROLES[role]]
    def obj(role):
        return _decode(raw(role))
    def digest_object(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    review = obj('actual_review')
    _require(review['status'] == 'GO_OBSERVATIONS_ONLY' and review['publication_authorized'] is False and review['full_code_cells_observed'] == _SHAPE[1] and review['actual_source_manifest']['sha256'] == hashlib.sha256(raw('actual_manifest')).hexdigest(), 'Actual independent review identity changed')
    _require(raw('before_notebook.ipynb') == checked['before_bytes'], 'Actual source book does not equal exact checkpoint')
    runtime_inputs = obj('input_manifest')
    # Physical target necessarily changed; validate the accepted before pin through
    # the exact strict inverse instead of pretending its old raw SHA is current.
    book_pin = runtime_inputs['source_pins'][_TARGET]
    _require(_same(_record(checked['before_bytes']), {'sha256': book_pin['sha256'], 'size_bytes': book_pin['bytes']}), 'Accepted runtime source book pin changed')
    before, after = checked['before'], checked['after']
    actual_book = obj('partial_notebook.ipynb')
    _require(len(actual_book['cells']) == _SHAPE[0] and _same({k:v for k,v in actual_book.items() if k != 'cells'}, {k:v for k,v in before.items() if k != 'cells'}), 'Raw notebook metadata/cohort changed')
    indices = [entry[0] for entry in _SIGNATURES]
    for index in set(range(_SHAPE[0])) - set(indices):
        _require(_same(actual_book['cells'][index], before['cells'][index]), 'Raw unrelated cell changed')
    transport = [_decode(line) for line in raw('raw_transport.jsonl').splitlines()]
    _require(len(transport) == review['raw_transport_rows'] == 208 and all(type(row) is dict and set(row) == {'monotonic', 'kind', 'cell', 'value'} for row in transport), 'Raw transport cohort/fields changed')
    _require([row['monotonic'] for row in transport] == sorted(row['monotonic'] for row in transport), 'Raw transport order changed')
    result, completion, exit_record = obj('result.json'), obj('completion.json'), obj('controller_exit')
    _require(result['status'] == 'FINAL_RECEIPT_PENDING_COMMIT' and completion['status'] == 'FULL36_CPU_REFERENCE_OBSERVED' and exit_record['status'] == 'CONTROLLER_EXIT_OBSERVED', 'Wrong actual result/completion/exit state')
    _require(result['cohort_indices'] == result['attempted_indices'] == indices and result['unattempted_indices'] == [] and len(result['observations']) == len(indices), 'Partial actual result cohort')
    _require(completion['result_sha256'] == exit_record['result_sha256'] == hashlib.sha256(raw('result.json')).hexdigest() and exit_record['completion_sha256'] == hashlib.sha256(raw('completion.json')).hexdigest(), 'Actual completion/exit linkage differs')
    _require(type(exit_record['observed_exit_code']) is int and exit_record['observed_exit_code'] == 0 and completion['requires_controller_exit_code'] == 0 and exit_record['normal_cleanup'] is True and exit_record['reaped_exit_code'] == 0, 'Actual controller did not complete cleanly')
    for record in (result, completion):
        _require(all(record[key] is False for key in ('retry', 'scientific_acceptance', 'human_acceptance', 'publication_authorized')), 'Actual scope/claim changed')
    adoption, claim, authority = obj('runtime_adoption'), obj('consumed_claim'), obj('authority.json')
    _require(adoption['status'] == 'ADOPT_ONE_FINITE_RUNTIME_ATTEMPT' and adoption['retry'] is False and adoption['publication_authorized'] is False and claim['status'] == 'CLAIMED_UNKNOWN_CONSUMED' and claim['retry'] is False, 'Missing exclusive consumed runtime authority')
    _require(_same(claim['authority'], authority) and _same(claim['budget'], adoption['budget']) and claim['budget']['max_kernels'] == 1 and authority['adoption_sha256'] == hashlib.sha256(raw('runtime_adoption')).hexdigest() == exit_record['adoption_sha256'], 'Actual adoption/claim correlation differs')
    _require(receipt['claim']['claim_id'] == hashlib.sha256(raw('consumed_claim')).hexdigest(), 'Derived claim differs from actual consumed claim')
    identity = obj('owned_kernel_identity.json')
    _require(_same(identity, obj('cleanup_owned_identity.json')) and receipt['cleanup']['kernel_id'] == identity['kernel_id'], 'Actual owned kernel identity differs')
    cleanup = result['cleanup']
    _require(cleanup['normal_cleanup'] is True and cleanup['reaped'] is True and type(cleanup['returncode']) is int and cleanup['returncode'] == 0 and cleanup['errors'] == [], 'Actual cleanup unsuccessful')
    started = [row for row in transport if row['kind'] == 'KERNEL_STARTED']
    reaped = [row for row in transport if row['kind'] == 'OWNED_KERNEL_REAPED']
    shutdown = [row for row in transport if row['kind'] == 'SHUTDOWN_REPLY']
    _require(len(started) == len(reaped) == len(shutdown) == 1 and _same(started[0]['value'], identity) and _same(reaped[0]['value'], {'pid': identity['pid'], 'returncode': 0}) and _same(shutdown[0]['value'], cleanup['shutdown_reply']), 'Actual shutdown/reaping transport differs')
    export = obj('actual_export')['rows']
    witness = obj('projection_witness')['cells']
    _require(len(export) == len(witness) == len(observations['rows']) == len(indices), 'Actual export/projection cohort changed')
    for ordinal, (index, observed, mapped, projected) in enumerate(zip(indices, observations['rows'], export, witness, strict=True), 1):
        tag = f'cell_{index:03}'
        # Paths are already a fixed source-pinned finite map, not caller inputs.
        run_prefix = _ACTUAL_ROLES['partial_notebook.ipynb'].rsplit('/', 1)[0]
        attempt_bytes = captured[run_prefix + '/' + tag + '.attempt.json']
        cell_bytes = captured[run_prefix + '/' + tag + '.after_at_reply.json']
        attempt, cell = _decode(attempt_bytes), _decode(cell_bytes)
        _require(attempt['evidence_class'] == 'REAL_JUPYTER_FULL36' and _same(cell, attempt['after_cell']) and _same(cell, actual_book['cells'][index]) and _same(cell, observed['raw_after_cell']), 'Actual raw whole-cell provenance differs')
        _require(mapped['index'] == projected['index'] == observed['index'] == index and mapped['id'] == projected['id'] == cell['id'] and mapped['actual_count'] == projected['execution_count'] == cell['execution_count'] == ordinal, 'Actual export/count position differs')
        _require(_same(_record(cell_bytes), {'sha256': mapped['actual_after_pin']['sha256'], 'size_bytes': mapped['actual_after_pin']['bytes']}) and _same(_record(attempt_bytes), {'sha256': mapped['attempt_pin']['sha256'], 'size_bytes': mapped['attempt_pin']['bytes']}), 'Export raw pins differ')
        _require(mapped['before_whole_sha256'] == digest_object(before['cells'][index]) and mapped['outputs_sha256'] == digest_object(cell['outputs']) and mapped['copy_only_fields'] == ['execution_count', 'outputs'] and mapped['preserve_canonical_typed_source_and_all_other_fields'] is True, 'Actual export slot authority differs')
        _require(_same(projected['raw_outputs'], cell['outputs']) and _same(projected['published_outputs'], after['cells'][index]['outputs']) and _same(projected['changed_output_fields'], observed['projection']['output_fields']) and _same(projected['source_projection'], {'canonical': before['cells'][index]['source'], 'raw': cell['source']}), 'Published exact typed witness differs')
        parent = observed['request_id']
        local_io = [(position, row['value']) for position, row in enumerate(transport) if row['kind'] == 'IOPUB_RAW' and row['cell'] == tag]
        _require(_same(attempt['messages'], [message for _,message in local_io]), 'Per-cell raw transport transcript differs')
        matching = [(position, message) for position, message in local_io if message.get('parent_header', {}).get('msg_id') == parent]
        shell = [(position, row['value']) for position,row in enumerate(transport) if row['kind'] == 'SHELL_RAW' and row['cell'] == tag and row['value'].get('msg_type') == 'execute_reply']
        completed = [(position,row['value']) for position,row in enumerate(transport) if row['kind'] == 'SHELL_COMPLETED' and row['cell'] == tag]
        _require(len(shell) == len(completed) == 1 and _same(attempt['reply'], observed['shell_reply']) and _same(shell[0][1], observed['shell_reply']) and _same(completed[0][1], observed['shell_reply']) and _same([message for _,message in matching], observed['iopub']), 'Actual raw request/reply/output correlation differs')
        _require(shell[0][0] < completed[0][0] and max(position for position,_ in matching) < completed[0][0], 'Raw shell completion order differs')
        expected = {'index': index, 'id': cell['id'], 'parent_msg_id': parent, 'execution_count': ordinal, 'source_sha256': hashlib.sha256(cell['source'].encode()).hexdigest(), 'whole_after_sha256': digest_object(cell), 'outputs_sha256': digest_object(cell['outputs'])}
        _require(_same(expected, result['observations'][ordinal-1]), 'Actual result observation differs from raw source-correlated objects')

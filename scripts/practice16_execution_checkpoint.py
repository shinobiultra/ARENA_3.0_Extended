"""Finite 1.6 actual checkpoint adapter; file validation only, never executes cells."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from scripts.rome_runtime_metadata_checkpoints import _json, _read, _record, _typed
from scripts.practice06_execution_checkpoint import _check_projection

_ADOPTION = {'contract_path': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/contract.json',
 'record': {'sha256': '4a8885daf73acacfdc715d0bfcc894ee4c1f198f9d474dc4b1c677019bdca856',
            'size_bytes': 116721}}

_TARGET = 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb'

_IDENTITY = {'chapter': 'chapter1_transformer_interp',
 'exercise_dir': 'part6_frontier_ml_infrastructure_practice',
 'page': '40_[1.6]_Local_Frontier_ML_Infrastructure_Practice',
 'section': '1.6'}

_MEMBERS = {'after': 'after.json',
 'before': 'before.json',
 'observations': 'observations.json',
 'receipt': 'receipt.json'}

_SCOPE = 'actual_1.6_independently_adopted_execution'

_SCHEMAS = {'contract': 'arena.practice16.execution-publication.contract.v1',
 'observations': 'arena.practice16.execution-publication.observations.v1',
 'receipt': 'arena.practice16.execution-publication.receipt.v1',
 'sidecar': 'arena.practice16.execution-publication.sidecar.v1'}

_SHAPE = (116, 42)

_BEFORE_RECORD = {'sha256': '4066bc77aa6c2038a75474c24986b9ac9b7f5690b5bd0641d3d0be46913f0d2e', 'size_bytes': 126531}

_SIGNATURES = [(4,
  'foundation-6e81315fc93dacceed8f2a8d5c440e2a',
  'ba9d07cb8c89198a5f61d8699cc6285fb65b94c25db885430e67ac53e21935ed'),
 (6,
  'foundation-9728d23e4b8b3027627e6916f1beac42',
  'aa9cf0f886e974369cac6187553724cbf4a2c745d21d887a1dffa64bb4e7c2c6'),
 (7,
  'foundation-7c2546e177b8801eedc972f21fb6b569',
  '6b9459fbd9a6742ef4077b2f2dacd852260c257659a0fc28b52cde15064d2871'),
 (8,
  'foundation-e2ecc9164d1238028e0e19ba0a1bc88d',
  '8d13653337d401a2c82c3a408666c68c0aa424483df4fe792edcb59baebba114'),
 (9,
  'foundation-5797cf063cd531cd4828ef16c7cd4f5a',
  'fc5b036b49ff5cb709938865fa1dd7d00943b500da36d6264334af5f899b0a52'),
 (10,
  'foundation-fe423287b62f7ac96487d0d257b6feca',
  'a0f878a3cc89b2abd9bb19095a4509a2c3952101f169d6ce4c1e05a6f856c29f'),
 (11,
  'foundation-6be94766722470c9f573f6fad8ebc414',
  'e60578cba8c556b7003f0e07b21c711ff9e01ef0432e1cb3100b2da4988c6caf'),
 (14,
  'foundation-de05eb5077569cc92c74092a8d540654',
  '9a2b80714deb1f8019357c59f61ac95b482f05792253b2b234a73d5350a569e3'),
 (15,
  'foundation-4405f1f343c8d4f7b73bdb74d39096d9',
  '038aa96feca083eff9907c13dfa374172aa19a402cba91e181d8b279768d5a99'),
 (16,
  'foundation-e2daf07c208bc754ecf16775a0b84b32',
  '22954f750ccd81eb769eb84b3ce0dfc93c881b2d37cc6bc8c9bd1703e619e99d'),
 (17,
  'foundation-2e701d2d9ec823ac937cc5f3b4718c10',
  'dd044bd613156eb829b0199bafb2626fa9362f53cf73b0439f95301ffa8d7198'),
 (21,
  'foundation-a4c6866e61062f776fdfafcd5f91a03d',
  'd13d2cf72f92011518b3e00e321061ae78f625caa8f8543ebc0cf36c15a2f675'),
 (22,
  'foundation-59b05881d4b4fb298db64a6c95cba669',
  '748822b2a7d410746a11d060d4b6287126b157a0907bd977ccff099801f6f0d0'),
 (23,
  'foundation-8abb8b43de5fafdd4c109bb71f22460d',
  'a13d094e4c1f96cdda7c4c1bf7cb3a07bc2bce2c65cd8123b05c43b48e2e1b1b'),
 (24,
  'foundation-e3a20f0856491e321e582a086dbe08f0',
  'b93b9183062c47b1bef7877f760cdfcb7e72e56afc897dcbd9a27a2657f65d6e'),
 (25,
  'foundation-143452f090d0bf9964cb8ed167840f09',
  '2788de77d0ab9eb08520121e4f66377abb7c9232b90f4cdc60fe008ece322e6c'),
 (27,
  'foundation-0972d2784032853b084ce90a655ba0cb',
  '0f8ea2b4520bb05c90cef0b7c8cef9ce9c90cd1ad47510f58dfb9b3fe44d65a8'),
 (31,
  'foundation-e9be41c3916516d6454d593e63555a85',
  '5a5ad03dd0c9c8113d447de035762f27ee79a94a8b0d20aadae7011eba74cebe'),
 (32,
  'foundation-813bf46b629c6b0b41ee26ad3b877052',
  'd92f17e187b6dfc7438b1326706c6e7c8a3ec2c187b554512d82b80c29776af2'),
 (33,
  'foundation-ff3bf9f0433e9a8227bc8e6b72a6dbd7',
  'c1338634a61e69d23eeecc45fa8970480da4a70c16f310149bd586cd771dbfdc'),
 (34,
  'foundation-bf23a30b98dd147e841dbc5f848ef5a6',
  '39b8b7b3fbd6b74b3b51f6ce221fe026fa59eab60e549773f6351ef4aefeeae6'),
 (35,
  'foundation-87b7df5a47533c2c457eb9f3959954b3',
  '6ad8112e0e02ed703735c0bf772e3a16af454013f9f3b1f3b700d7d690094ae7'),
 (36,
  'foundation-1f433747877edb84a6975dbe21066be2',
  '21a0023fb0f0019245f3e2cff21f0569099e5f25b7593e56d4b06c66bcfa0d2d'),
 (37,
  'foundation-3d8cc05259cdcfc8b2d505360685cc53',
  '57319e65ae1b0f0924bd456e94e326c1637dca105d9188413f13934d5fdab161'),
 (38,
  'foundation-5b80a462bd5e5b7313e6be7b304b7c86',
  '9d8e51829d363c17de5ca2e4c9cad7c9ef29d032abddab1f06524ea5e0f04cfc'),
 (42,
  'foundation-1919419988acb6e11830b7c4785bb269',
  'ff04deb5917f5f0673210b03a81dc83ea3c97dc027dc0e4be949d93033e0c58c'),
 (43,
  'foundation-c6b1e3153ad7a8c2aef04739cdab6e7d',
  '4c2da20127f6f0e51e8e045ffce887285e3766ee36551ae417160a46fbfe3e7e'),
 (44,
  'foundation-897e4ce5d2752a5bea54dfe4d636d4e9',
  'a3b606945df085628e3b7b7a90a294212258dd422755f64f1587433120abe180'),
 (48,
  'foundation-59cc3fee73dbc42ec04cd87a4eecccc0',
  'b52776cde5386ec737a71ad21912c2725ab594ea3af8553013c8d3ec1e84ce8e'),
 (49,
  'foundation-d91c725b4f12b7ee16ee017fe5fa8341',
  '82d4e86cf4a632b59a89f40477234efa9162d4af08b316b30c2f9e6ce44d8ea6'),
 (50,
  'foundation-e07cf3d09353026e22f572b8bb52f7a5',
  '6b0ade31fb88950ac6efb63b25738385785de8f8db1cfd9de101e9954afecb05'),
 (56,
  'foundation-f009c0d3bde918517c8665d80aa7f9c5',
  '6c94c855354f13bd6279366430e0d403cbebd3ee593d44d251eea0abf736a5fd'),
 (59,
  'foundation-a83854ba86f5a6f777418d00ae79935f',
  '06682077c67f6aca69b3955d5862ec8cb504aef703c16f26741fe29daee7d479'),
 (60,
  'foundation-129cc30147917c5e4f0d69b8740fcf8f',
  'adae25e479dacddc743304f4fc5a46cd72ab51b90c2e7003d934cbe152a744e4'),
 (101, '7e06ee88', '06c10aea1d8981cefb820fd701b308abb8b8b0eed6aa212d71054e1bad14beff'),
 (103, '83512b82', '0c8a2f885c9ce17804949f6f3daeef98953bd4cfa524573a5ba4057423b2d343'),
 (105, 'c652749d', '005ce74c308127ca1ac6274b6050aa62ec26675457b4dbc627880eef815d2fb4'),
 (107, '6e8bd3ff', '90a7411d8b8de81eaae2c9702f4dc949e87fc63a324ae172de8a6f097ca65ea0'),
 (109, '6a73d373', '7ef46e29618aaac868561c10558216b7f1fac18671c9978ed8c7b6416cab0e86'),
 (111, '625fc77a', 'a517584c85df177c9fa3d59c5d2646835f6214c24d32b05af236eff1129b1766'),
 (113, '2523bcd5', 'f535856c4643d5b37737e09f950f5b1e899e31c3063181a2dbbb9bc1e37137e7'),
 (115, '9ab24c81', 'c6e3241b6d6b5b519ea55a23167d5068c7ed85a074d015be38e3522088b02300')]

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
 '.venv/lib/python3.14/site-packages/safetensors/torch.py': {'sha256': 'f3f476d1f8c04fe65fa3797426556a0b7afa43f8c4db9db6b799c7cf84748f3d',
                                                             'size_bytes': 20312},
 '.venv/lib/python3.14/site-packages/torch/__init__.py': {'sha256': 'd9dfff4b75d46e4c75572200a3466b70231d05b0318e38ac1bd121789165fb49',
                                                          'size_bytes': 109312},
 '.venv/lib/python3.14/site-packages/torch/version.py': {'sha256': '38fbf6a65805efde37fb1773c0ce4eeceeb29cefb5895e6b1690109d0bbbe6a6',
                                                         'size_bytes': 317},
 '.venv/lib/python3.14/site-packages/torchvision/__init__.py': {'sha256': '775cb7a87f9b26eb9fcb498f79313560a4878968d458acaf95b9b5ae6b4656d3',
                                                                'size_bytes': 1969},
 '.venv/lib/python3.14/site-packages/torchvision/version.py': {'sha256': 'fbb3cd704fc4320d289c2e9122c9c769ae2f7191c39c0fbe910704b24263835d',
                                                               'size_bytes': 203},
 '.venv/pyvenv.cfg': {'sha256': 'd933e3b78120f8d4ac9acc041ff95d9c31e9a405d06a0d18992ffd627eb6d4f5',
                      'size_bytes': 202},
 'arena_ext/__init__.py': {'sha256': '98abfe2501c14bb586c6e5711446dd613a44fde9c7e0fc1b6d1438a986e58b4e',
                           'size_bytes': 2917},
 'arena_ext/activation_store.py': {'sha256': '574fd258a518800b31771fb6805b347c083a165a328bd1a6769632f5690bf23b',
                                   'size_bytes': 3530},
 'arena_ext/environment.py': {'sha256': 'bf8c93a4a23411b1a56b5576f21d8b3111e0fba8d4a46c903a350c297f4b4fac',
                              'size_bytes': 3519},
 'arena_ext/hf_parity.py': {'sha256': '5c48555bdafbcb849cc83263a0e2743d0950f105fb21ffd2ffdebf585d01b829',
                            'size_bytes': 3865},
 'arena_ext/memory_budget.py': {'sha256': '1a118a19a12a494cf738605fca99abfbbe82ec7e31f4bf517e1e923a59caa802',
                                'size_bytes': 3672},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb': {'sha256': 'c65c73ecda6457d426f934f4801a9f018972d4e17f73d78bca82a323449ff034',
                                                                                                                                 'size_bytes': 14460},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/artifacts.lock.yml': {'sha256': '8139c790fea389b0d768d7ff356f8c249ebbe432ae6981063f64e6f3644fb30d',
                                                                                               'size_bytes': 2799},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/expected_outputs/local_frontier_infra_signature_result.svg': {'sha256': '04dd2300792ca0100da14ab40472cb1ca6e526f038a1f43b3f5b0ab991561b41',
                                                                                                                                       'size_bytes': 2085},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/expected_outputs/local_frontier_infra_validation_loop.svg': {'sha256': '928933bd429be8dbd8ef8d74994588127880b02b0b63881caa0b340772d7d2bd',
                                                                                                                                      'size_bytes': 3361},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/solutions.py': {'sha256': 'f11ebeae7499b24fa8b997fde685beef5a1025916545e0d47c39d11ed012a082',
                                                                                         'size_bytes': 7606},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/tests.py': {'sha256': '72f9946b64532346ea7e159d10b773b95ddea5681939d0e014714125224a75cf',
                                                                                     'size_bytes': 10554},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/utils.py': {'sha256': '11beb3f7d548c52d99532eebed81b50607f5f7712a1f2745314cbe3b9e399cb4',
                                                                                     'size_bytes': 462},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/verification_report.json': {'sha256': '7c928247d424697361ca1d85b168c0e53741a05d59c79256fbb7ea6f23477166',
                                                                                                     'size_bytes': 10159},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/verification_report.schema.json': {'sha256': 'ecd833883885077ad5d7a0e239d984d01462255e94266212cf195ab4d4e27b3d',
                                                                                                            'size_bytes': 2669},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_exercises.ipynb': {'sha256': '445c5099bdfb18408a674f65d6bd114c9d4cc5282e5d1e43111e608a2740189e',
                                                                                                                                                   'size_bytes': 133389},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb': {'sha256': 'd8b0125526a3cef24952ae2b883bfda1eac5d02a687641a5dcf6bb82fd56928d',
                                                                                                                                          'size_bytes': 124027},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/README.md': {'sha256': '64348b1dd337061a78ced4813d6a5ebcd4b93229333f3740843088e8338ae6ae',
                                                                                               'size_bytes': 2611},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/artifacts.lock.yml': {'sha256': '8139c790fea389b0d768d7ff356f8c249ebbe432ae6981063f64e6f3644fb30d',
                                                                                                        'size_bytes': 2799},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/expected_outputs/local_frontier_infra_signature_result.svg': {'sha256': '04dd2300792ca0100da14ab40472cb1ca6e526f038a1f43b3f5b0ab991561b41',
                                                                                                                                                'size_bytes': 2085},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/expected_outputs/local_frontier_infra_validation_loop.svg': {'sha256': '928933bd429be8dbd8ef8d74994588127880b02b0b63881caa0b340772d7d2bd',
                                                                                                                                               'size_bytes': 3361},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/media/local_frontier_infra_signature_result.svg': {'sha256': '04dd2300792ca0100da14ab40472cb1ca6e526f038a1f43b3f5b0ab991561b41',
                                                                                                                                     'size_bytes': 2085},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/media/local_frontier_infra_validation_loop.svg': {'sha256': '928933bd429be8dbd8ef8d74994588127880b02b0b63881caa0b340772d7d2bd',
                                                                                                                                    'size_bytes': 3361},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/solutions.py': {'sha256': '383f5d82b92b542d7f55bbe73ebd480c0b1b5701174e8b74a1d485c13a203a38',
                                                                                                  'size_bytes': 24054},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/tests.py': {'sha256': '72f9946b64532346ea7e159d10b773b95ddea5681939d0e014714125224a75cf',
                                                                                              'size_bytes': 10554},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/utils.py': {'sha256': '11beb3f7d548c52d99532eebed81b50607f5f7712a1f2745314cbe3b9e399cb4',
                                                                                              'size_bytes': 462},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/verification_report.json': {'sha256': '7c928247d424697361ca1d85b168c0e53741a05d59c79256fbb7ea6f23477166',
                                                                                                              'size_bytes': 10159},
 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/verification_report.schema.json': {'sha256': 'ecd833883885077ad5d7a0e239d984d01462255e94266212cf195ab4d4e27b3d',
                                                                                                                     'size_bytes': 2669},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before.tar.gz': {'sha256': '03a9f7c06b19982a776b474bd86aee12967421700cf30712c4aff29f7d5e42e0',
                                                                          'size_bytes': 4724306},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before.tar.gz.review-bundle.json': {'sha256': 'ecc91b2885712c48f1d7545809a77bacbc7707958dff15a916df2c399554d876',
                                                                                             'size_bytes': 3997},
 'docs/mastery/evidence/lesson_integration_20261021_q008/before_manifest.json': {'sha256': '1b15fca79d95a98f0cb2efb94350f4e7badba748bf64590999248751c07a1c54',
                                                                                 'size_bytes': 2380},
 'docs/mastery/evidence/lesson_integration_20261021_q008/navigation_inverse.json': {'sha256': '426ce9753c3084e44839262c27032f420a3ee4adbdf40c831d4a054bc83a56a8',
                                                                                    'size_bytes': 18533},
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
 'infrastructure/chapters/chapter1_transformer_interp/master_1_6.ipynb': {'sha256': '10838f1f5675899c60572d6c509c7033cd97f525cdf8f378a0a614f74a3a7bd9',
                                                                          'size_bytes': 127287},
 'infrastructure/chapters/chapter1_transformer_interp/master_1_6.py': {'sha256': '237fe5edc969e23ece5440cbb7bed04a5ef484615b3a63ea0962f8c627f1077a',
                                                                       'size_bytes': 94601},
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

_PROVENANCE = {'docs/mastery/evidence/practice16_execution_publication_20261023_v1/bundle.tar.gz.review-bundle.json': {'sha256': '69df07f8a63403ca82f055e22afe8bf179fe64b637ed44445722da391a015993',
                                                                                                         'size_bytes': 1851},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/canonical06_transition_receipt.json': {'sha256': '25d3badcfeb1441368f32ee6c7c7c472d4f5fb3bf02096d744aa3da17ce8b720',
                                                                                                            'size_bytes': 35550},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/prospective_current_q008_source.py': {'sha256': 'd1c841418f9c3ba8bb94f6dc9c807cb59902bb805ed67a9006471fef6aa96a54',
                                                                                                           'size_bytes': 7229},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/q008_transition_witness.json': {'sha256': 'f2166c7705bfedcbb1036c8e5c318dff910c01e6f6512c63fe27c2f98444fede',
                                                                                                     'size_bytes': 1060},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_dispatch/actual_controller_exit.json': {'sha256': 'cc63c40f9edd9a0aacf916290da137129f5d4a78d0c64c092448135879211453',
                                                                                                                                    'size_bytes': 252},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_dispatch/root_preexecution_adoption.json': {'sha256': 'ac98f8f16c94cdace80331748b1087c548b0201bab850cd97d2104ed9da8d4cd',
                                                                                                                                        'size_bytes': 1098},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/FROZEN.json': {'sha256': 'aedcf76dace30bc5ae008452a8b27ef439d4910060572c0605b7e5a4550059b7',
                                                                                                                          'size_bytes': 742},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/attempt_claim.json': {'sha256': '9c73929527038a576a457b94a6850f44e4935ecab79db043ced8595f951e413f',
                                                                                                                                 'size_bytes': 754},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/input_manifest.json': {'sha256': 'ece627714995744d20aac3b6635af8e9446a2af121b8aa45621c8b1aeafadc80',
                                                                                                                                  'size_bytes': 81872},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/inputs/tests/test_q008_practice_projection.py': {'sha256': 'd6c0e5ce3914735a48412cba1679af0fa67980cc0501fee989afc5b55b2558d5',
                                                                                                                                                            'size_bytes': 7134},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/jupyter_adapter.py': {'sha256': '25c2d4fe724968139637cfbd39527c5d0449a7c21e24bcdcbb3b4014fc01ea14',
                                                                                                                                 'size_bytes': 14916},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/packet_manifest.json': {'sha256': '512c5e8fcecb08fb57587ff0716a828dc121db0237e677ccb99b349fee020735',
                                                                                                                                   'size_bytes': 19136},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/authority.json': {'sha256': '31a80d4d19873c74cb7a41d794f455f13f01702efae98b56ae704a6e223a38aa',
                                                                                                                                      'size_bytes': 561},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/before_notebook.ipynb': {'sha256': '4066bc77aa6c2038a75474c24986b9ac9b7f5690b5bd0641d3d0be46913f0d2e',
                                                                                                                                             'size_bytes': 126531},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_004.after_at_reply.json': {'sha256': '0053bbfe395f323aa497fc0a82a0551c2c80343666470b036b42d71ae9085e72',
                                                                                                                                                    'size_bytes': 313},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_004.attempt.json': {'sha256': '8f75b86ae2f6c916f26a424dd71f443a172d3e31890f2241c402675727c8e05f',
                                                                                                                                             'size_bytes': 4718},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_006.after_at_reply.json': {'sha256': '7b51069a7656571f5f83dffa1d10802f1432be45abe6b8c9dfec7b9fac72273d',
                                                                                                                                                    'size_bytes': 1406},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_006.attempt.json': {'sha256': '0e7f5ddc790c31d2738064d947ad97877d33fc5fca374fb83d3718333b7303a9',
                                                                                                                                             'size_bytes': 5590},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_007.after_at_reply.json': {'sha256': 'a468f1468eb5f2c604f6c0d8c0521abbf1f09ceaa0e4a5c97fdb261db0512a09',
                                                                                                                                                    'size_bytes': 367},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_007.attempt.json': {'sha256': '95118d1c39cfe3a273d3fc9e2755ea67cd729aeb130b0e3641da6b5cf239c368',
                                                                                                                                             'size_bytes': 3512},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_008.after_at_reply.json': {'sha256': '6753ee989f49b1fba8736274f35df35ba5e2798746d8fa13e19201d2ac794d12',
                                                                                                                                                    'size_bytes': 1123},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_008.attempt.json': {'sha256': 'f0fce1610fbfd104769e7c1e723c22f928ac5fda6037964937223948ae489cdc',
                                                                                                                                             'size_bytes': 5024},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_009.after_at_reply.json': {'sha256': '4fa64c9498ded5142c4879d19eefda01ffa6455e8d53f065778d0b9e798dea43',
                                                                                                                                                    'size_bytes': 1266},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_009.attempt.json': {'sha256': '73169eb9f704aaeb46947aac9300016f349c0a8f0238016497e2fb3d60891a0f',
                                                                                                                                             'size_bytes': 5310},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_010.after_at_reply.json': {'sha256': '63368707414e351a4a8ff2c98f063ad55f18c94ed318b460964f58a6156a02b4',
                                                                                                                                                    'size_bytes': 2006},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_010.attempt.json': {'sha256': 'e09a14511871b666a8b10fb86d2a9f1618306a3bcfd604bc6f1139b5741e2dca',
                                                                                                                                             'size_bytes': 6790},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_011.after_at_reply.json': {'sha256': '344a12c0e6cd5e022f2d84087140018773138c8f1dc1cbbca27f72650e19dd5d',
                                                                                                                                                    'size_bytes': 381},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_011.attempt.json': {'sha256': '6cea523b19a4fc47cff5c5cad2c77c0e9f57ff0cc2c24369f78622b525b6f385',
                                                                                                                                             'size_bytes': 3540},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_014.after_at_reply.json': {'sha256': 'f2a74e62ad52d014a523e43973e66cc8205d1636aedcb3d23aa3201184633dd8',
                                                                                                                                                    'size_bytes': 221},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_014.attempt.json': {'sha256': 'bd7d4d85c8449521c0e0662bbb8ab4cd3138b3d08f5c8accdbc545f27d780d60',
                                                                                                                                             'size_bytes': 3220},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_015.after_at_reply.json': {'sha256': 'b34139c0495b9a96ee40dee80eeda3b6bfaa1a34c06a2281c67e490fdf41129e',
                                                                                                                                                    'size_bytes': 1141},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_015.attempt.json': {'sha256': 'd01cadc9877b84a97fc54389784fbe33c060f076c2fbf7619aadd11da5e9454a',
                                                                                                                                             'size_bytes': 5064},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_016.after_at_reply.json': {'sha256': '09e90c81fb6845636b99ccfa387e15266be83f87da9f96d57d063deb746ada0a',
                                                                                                                                                    'size_bytes': 580},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_016.attempt.json': {'sha256': 'f52a3451e0f9b9837fc0dbd79d474b23b7b3ea0a2e7216f988bbbe39315dc1c6',
                                                                                                                                             'size_bytes': 3943},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_017.after_at_reply.json': {'sha256': 'de5c15f23bbb838349ab8434279dd402839f6982074aa0258896ff547471b20d',
                                                                                                                                                    'size_bytes': 560},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_017.attempt.json': {'sha256': 'c2a133c4a9731f8effe553ea2060476d81dd4e2d6668abd1422b3109947b94ba',
                                                                                                                                             'size_bytes': 3903},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_021.after_at_reply.json': {'sha256': '43072c79e90a5b7dd4184454de559b4f6741099b999978c951f34d8df64a8526',
                                                                                                                                                    'size_bytes': 464},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_021.attempt.json': {'sha256': '6d3c826853c2d54b0a95f1121231bb97e15053f4184cd59e8d4dfd546227e290',
                                                                                                                                             'size_bytes': 3711},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_022.after_at_reply.json': {'sha256': '8be5acc86ce10215f6686d94df20268b3dfdbee96b34390530d652594fd31e4b',
                                                                                                                                                    'size_bytes': 1666},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_022.attempt.json': {'sha256': '90911ee30573ac50c0919b257be431e407d34145eb258a9fb8cae1c640df91ad',
                                                                                                                                             'size_bytes': 6115},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_023.after_at_reply.json': {'sha256': 'ab423d023a4ed84aeb20c11a1f6480c72b89e0678767d66f9381c4b1cdab614c',
                                                                                                                                                    'size_bytes': 687},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_023.attempt.json': {'sha256': 'a5771155683715eb0b83665e3c96436b1389d01e0b5086da5d6b92c08b4c6156',
                                                                                                                                             'size_bytes': 4157},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_024.after_at_reply.json': {'sha256': '862ca6d31e4781485a4eed56abbfbd59bcd66367f728bf2a2545607cbe29e8d1',
                                                                                                                                                    'size_bytes': 572},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_024.attempt.json': {'sha256': 'c0e491d222a4b00868a9f6c0204a6148ee826a139c4228735b6c162fa845b446',
                                                                                                                                             'size_bytes': 3927},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_025.after_at_reply.json': {'sha256': '6e612029feadb99ebd2f70599ec3460b9fa390dbe112969bc1207291ee668c2e',
                                                                                                                                                    'size_bytes': 432},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_025.attempt.json': {'sha256': '4894336a2305fc5dd3a28c6007514440b8758dad82ac0979b654b178687f8874',
                                                                                                                                             'size_bytes': 4257},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_027.after_at_reply.json': {'sha256': 'e47f28cdfe6e9fc4cfaf0ea05e0d48cf41fa7d6267108f406b72e03dd8be787e',
                                                                                                                                                    'size_bytes': 687},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_027.attempt.json': {'sha256': '87397e19677c8bcbde2ecfe1e9f9347810b5a94e53cf880512538b98345c80e6',
                                                                                                                                             'size_bytes': 4157},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_031.after_at_reply.json': {'sha256': 'f617ded1f7bea42ac0d2ccd3f3dc76e1151f5d0f8b15967d2bdecaf74054b680',
                                                                                                                                                    'size_bytes': 595},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_031.attempt.json': {'sha256': '0ff8161998cbd05db0a8e3422ee9e5e93b14366ef8670d412035b14d31bed021',
                                                                                                                                             'size_bytes': 3973},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_032.after_at_reply.json': {'sha256': '3e43aea4fe18f57fc0284cd4fec87e57e13c2885c06eac985b3dad4d5445a151',
                                                                                                                                                    'size_bytes': 768},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_032.attempt.json': {'sha256': 'ced531fb15f8c4b7d008326749272f596ed10b432226aa13e1ef1f54f248b427',
                                                                                                                                             'size_bytes': 4319},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_033.after_at_reply.json': {'sha256': 'bd40d0a5c7aba8aa3025edb1f85d410d90f2278e886759059a3b99469b184075',
                                                                                                                                                    'size_bytes': 687},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_033.attempt.json': {'sha256': 'f58849000b9ea425dc72a772abc101a0b5ad1a0d114ca368c05d0043458be9e3',
                                                                                                                                             'size_bytes': 4157},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_034.after_at_reply.json': {'sha256': 'c1361f0e61643efbe45604f762bb5d0963f22707d905aceaa56eb7d6c53ccff1',
                                                                                                                                                    'size_bytes': 719},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_034.attempt.json': {'sha256': 'd0a1d69a4f1a10a649579592955badbc8688690cd9908d6fdf022ea977c8489f',
                                                                                                                                             'size_bytes': 4221},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_035.after_at_reply.json': {'sha256': 'c55e73598f397c7a4ee1991b4bc126aad2cb32d86681bc700d9ae7a1e88f0dff',
                                                                                                                                                    'size_bytes': 531},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_035.attempt.json': {'sha256': 'b472b5d3ea6dce137d905327a870fe960477f1a6b6528c90c36e0ccad6860155',
                                                                                                                                             'size_bytes': 3847},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_036.after_at_reply.json': {'sha256': '20be0ea1354362a547a96b681623687e62ed87c8bfcd132a1c80f79ac8223fb1',
                                                                                                                                                    'size_bytes': 595},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_036.attempt.json': {'sha256': '5f2b03d0fb09cd73bd7d507a509419fcfcedbb1e0736e2eb52cfbacf7d998df6',
                                                                                                                                             'size_bytes': 3981},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_037.after_at_reply.json': {'sha256': '532c58077c35af10d85ea2e2f2e8cf8133b1206bdb2d58226cc50f6b3a4632ff',
                                                                                                                                                    'size_bytes': 317},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_037.attempt.json': {'sha256': '46d9d910355453fc6ecd5ae8729dfbdae39d273467466de68a095c9102e21242',
                                                                                                                                             'size_bytes': 3425},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_038.after_at_reply.json': {'sha256': 'e1cdbd0b0d6904781e50a09ac44e72056c10f5723e2076a95755235eaf25c129',
                                                                                                                                                    'size_bytes': 978},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_038.attempt.json': {'sha256': 'ee0a99f7066b6d954dd542fc88be47d8c846bba51fc2b88bbdd48f14f6d214e0',
                                                                                                                                             'size_bytes': 5359},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_042.after_at_reply.json': {'sha256': '08c74bb87f68dbe5611dbae66e7a719d16284c26f1a961d4f834c651407f7872',
                                                                                                                                                    'size_bytes': 3285},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_042.attempt.json': {'sha256': '32e57e03ca734cc1470e6ee7283bec1542591f267de92ebc63680bdb530bac2d',
                                                                                                                                             'size_bytes': 9361},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_043.after_at_reply.json': {'sha256': 'a978292971d082af5a6e732a28e86fef181efc9181455aa0dd634cafd2c0321d',
                                                                                                                                                    'size_bytes': 727},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_043.attempt.json': {'sha256': '05dac843d4d06d3ffbcffcff5202398ce9ed1e39cffc71d58fc40a7a5b0c893e',
                                                                                                                                             'size_bytes': 4245},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_044.after_at_reply.json': {'sha256': '31e7274aa939481a6f8dcfb3c9968fb3034a48524fdf796d61e9cf7ed79a4daf',
                                                                                                                                                    'size_bytes': 533},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_044.attempt.json': {'sha256': '606d83a082f0098c62649440cf7827e6297f577333d3ea28c6194ed9bbec6c04',
                                                                                                                                             'size_bytes': 4469},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_048.after_at_reply.json': {'sha256': '6be043be8761a369191cd8123105b83df8c39c837107fbf3b4be552191bed752',
                                                                                                                                                    'size_bytes': 1008},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_048.attempt.json': {'sha256': '55398ca9ae8a9c7c4fed4c221d7b4ff46422a7bb0971e0eb0bb5a6dbb7fdfaa3',
                                                                                                                                             'size_bytes': 4807},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_049.after_at_reply.json': {'sha256': 'a6a9ea735d69566d763ca6fbd2b78629a9299722f2dbb9ba8a6391ec22ebf522',
                                                                                                                                                    'size_bytes': 783},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_049.attempt.json': {'sha256': 'ad7fc9891a612f3ad2ff455516def36e430a55342605841d42359d4b6173164e',
                                                                                                                                             'size_bytes': 4357},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_050.after_at_reply.json': {'sha256': 'aa416c95a69ae6cecc2d3a282191fadc4d72a5ab78a04cc81f537cc588941b04',
                                                                                                                                                    'size_bytes': 796},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_050.attempt.json': {'sha256': 'ba6d10a35c6a70a412a7c710a8ad208aad4b76ae73ce26daaea834c85652f798',
                                                                                                                                             'size_bytes': 4995},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_056.after_at_reply.json': {'sha256': 'db71b676f5875366a51914d9e1084344cc00a1a1c6215e8ce8c1582940ebbfac',
                                                                                                                                                    'size_bytes': 626},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_056.attempt.json': {'sha256': '40b0d198cdfc40fae701482d4b48d3d24b7ba30672b9ab03d6f839f93513dcb0',
                                                                                                                                             'size_bytes': 4655},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_059.after_at_reply.json': {'sha256': '72f4d4c5cc70fbcd6f7f172f2937e4e676f600ff59d3703abb21e4ec5d41dc8b',
                                                                                                                                                    'size_bytes': 3669},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_059.attempt.json': {'sha256': 'bb4271523d3ec960727da6b185a1578e809616b1f2e382ef4b7d542df9dd5d97',
                                                                                                                                             'size_bytes': 10129},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_060.after_at_reply.json': {'sha256': 'efd7bba05ddeeff90bb8e0ad28007b65477bd8743b181927031aa386ca1b426b',
                                                                                                                                                    'size_bytes': 310},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_060.attempt.json': {'sha256': 'dd2a31577d30ed8de39ce87ea2c66701bc46e77be517d33b6a1f354adb12e4f8',
                                                                                                                                             'size_bytes': 3411},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_101.after_at_reply.json': {'sha256': 'dfabe5c286edc59f6d9535e403fbf7e1c98a9a37b7efcd49743b757ac31eab28',
                                                                                                                                                    'size_bytes': 1452},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_101.attempt.json': {'sha256': '0989ad3e58731c177ba9623bde257b566de70248343277a8bbe524d6e468be85',
                                                                                                                                             'size_bytes': 5513},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_103.after_at_reply.json': {'sha256': '9a0132dc4e72691947b9063d9ee1d7c362fae18c57b96b6aa4d4a3a7d568bb89',
                                                                                                                                                    'size_bytes': 1237},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_103.attempt.json': {'sha256': '59478591b7454b9433eaa59101b8125476ddb2fe999eeb22c4da79c36ebd6cf9',
                                                                                                                                             'size_bytes': 6314},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_105.after_at_reply.json': {'sha256': '3b40e0ee23a2c77b5e86b8a2a503c52a80a7016ed83f41157e24583326a419c3',
                                                                                                                                                    'size_bytes': 934},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_105.attempt.json': {'sha256': 'd5cacb98108562d60f08590642d5e55c248b5b0ed8c3336b5641ede0e609538e',
                                                                                                                                             'size_bytes': 5089},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_107.after_at_reply.json': {'sha256': 'dce5fbbdb4f3679b4d081742688ef6b0be97e7e2f2d39f134cbaf1a9c9058eaf',
                                                                                                                                                    'size_bytes': 1031},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_107.attempt.json': {'sha256': '3af3dc9fbbab0f23436e0fcbc567a03481a3dee855d4a4190a229867362aa436',
                                                                                                                                             'size_bytes': 5283},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_109.after_at_reply.json': {'sha256': '1206482f8f2b1addc19d012072b8e66f613f60c13a6712ae092d1ac12f474f3d',
                                                                                                                                                    'size_bytes': 599},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_109.attempt.json': {'sha256': '28964521637f8b0425de972fec67a93280b0d9a432526d9507a372b6bedc10ea',
                                                                                                                                             'size_bytes': 4419},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_111.after_at_reply.json': {'sha256': '3b2350b845295053d4d7ad2811bb55fd41f078d3db21a37ef7ab05df37fd0e5f',
                                                                                                                                                    'size_bytes': 730},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_111.attempt.json': {'sha256': '4f71025cbc4cba1f411969cfdbf27b894925c685bd9287a107230ba98adbf910',
                                                                                                                                             'size_bytes': 4681},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_113.after_at_reply.json': {'sha256': '2b6c14b55fa86f0c2a562960bcef7a98ad5f13df757045deb3ddf9bdeeb44682',
                                                                                                                                                    'size_bytes': 2663},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_113.attempt.json': {'sha256': '472a5c1b174253b4ecb3ea4d3212bd6abbffff95eb06dd49ccd6d68bdf839c03',
                                                                                                                                             'size_bytes': 9166},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_115.after_at_reply.json': {'sha256': '5b8318442998e48786ff8f31a1be6469c363d2e67a07cd31aad929d94d05f417',
                                                                                                                                                    'size_bytes': 1268},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cell_115.attempt.json': {'sha256': '144a279675ffc356f4bd2e693f5aaa266e37c0c6abb707a1725618bf4072cb94',
                                                                                                                                             'size_bytes': 5757},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cleanup_owned_identity.json': {'sha256': 'f52c05608dc1d8559799c46f6667b93ef651474de456a9fe21294fbef096f854',
                                                                                                                                                   'size_bytes': 760},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/completion.json': {'sha256': 'bec74990b65f6a8938f6198d3cec349a6028647034a2bc2c6dfc3480eb49d21d',
                                                                                                                                       'size_bytes': 420},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/environment.json': {'sha256': '295f26f016e1023cf5fb06a070c909d525398ad84592a0a6d562da540db55022',
                                                                                                                                        'size_bytes': 2526},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/kernel_stderr.txt': {'sha256': '3f2b766f08dfd18a853f26878cf7e0d8cad05ab72687a30d2e374da957f6a5a5',
                                                                                                                                         'size_bytes': 283},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/kernel_stdout.txt': {'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                                                                                                                                         'size_bytes': 0},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/namespace_observation.json': {'sha256': '9b743cc53bf3ab2055e15ca7569c17b9d6acb0c4d4f0b50722999c67c2e7c045',
                                                                                                                                                  'size_bytes': 1363},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/owned_kernel_identity.json': {'sha256': 'f52c05608dc1d8559799c46f6667b93ef651474de456a9fe21294fbef096f854',
                                                                                                                                                  'size_bytes': 760},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/partial_notebook.ipynb': {'sha256': 'ff6085fec717b4e44303addf10cc69680af3d5c6a2edc77ddcc4f3586242ee46',
                                                                                                                                              'size_bytes': 109674},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/progress.json': {'sha256': 'ea8408466d0f2603c9691a5ae24aff5d1894c659867e04cda6c31a58c4aeae67',
                                                                                                                                     'size_bytes': 17100},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/raw_transport.jsonl': {'sha256': '484cbba30df0602810fa1d287349b3f540263d4ae8d995e19a0d7d60ca7dab59',
                                                                                                                                           'size_bytes': 222716},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/result.json': {'sha256': 'bdfdb2428dde72ef061e7b3b4e27d8cac8905033976095bb97e0293ba6f71fa2',
                                                                                                                                   'size_bytes': 18174},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/state/ipythondir/profile_default/startup/README': {'sha256': '638ec6b584245b2ea977392aca550d2fa7814993142111b04f15cf505fc10417',
                                                                                                                                                                       'size_bytes': 371},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/terminal_pending.json': {'sha256': '13f111363d041b5fa02acf4db9e8ca28c7e983d7cf14356efa5f374f3fbeb4af',
                                                                                                                                             'size_bytes': 538},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/whole_before_code_objects.json': {'sha256': '38b262821008463a247a91e8009c7eb6bb51888d39b3fcb59efc879f1e7a6643',
                                                                                                                                                      'size_bytes': 42182},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_practice16.py': {'sha256': '4d1192b96f0b7bc7a6da11c186dd75138d7ea1e08f97d4f1effec5c4f2889450',
                                                                                                                                'size_bytes': 22620},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_review_v1/FROZEN.json': {'sha256': 'a6b977736c9f0e84e7713ed97b2c995e0d5c47014dc21aa9bdd3d9ff4b992342',
                                                                                                                     'size_bytes': 355},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_review_v1/review.json': {'sha256': '16bc63fec1a031f447a262186fc7f0160b2604bd3b7b8e020ca6d64f9f2b4769',
                                                                                                                     'size_bytes': 1404},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_review_v1/review_manifest.json': {'sha256': 'be3890988c13a0f1c655a2d29a8a171969a4a5b70b56a7216cec4eeeac25a609',
                                                                                                                              'size_bytes': 31760},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/FROZEN.json': {'sha256': '4235f703be8dee7828af8f6d7b202a90241e3d18d84f8cd263e4c695943b0cca',
                                                                                           'size_bytes': 1022},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/REVIEW.md': {'sha256': '3121df1e600965f77ba95dd0ce72ae0bea9f81f093c0ef047f14cbcdd3751a2e',
                                                                                         'size_bytes': 6300},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/actual_source_manifest.json': {'sha256': '64421bb6e710c81bc587cd43037016982dce108bbdb3e2fb5fa64f69e861f9e5',
                                                                                                           'size_bytes': 82879},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/observed_slots.json': {'sha256': 'c86a8f8998ac30352c247832382a5ecacef5200ad2a09586d2d4006eb76c58f0',
                                                                                                   'size_bytes': 36632},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/verification.json': {'sha256': '143e4c56789813e31a78d4f0310800a850bd2360120782916ee3bc01d08c8f28',
                                                                                                 'size_bytes': 1840},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/verification.log': {'sha256': '143e4c56789813e31a78d4f0310800a850bd2360120782916ee3bc01d08c8f28',
                                                                                                'size_bytes': 1840},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/verify.py': {'sha256': '4a9e0a2a0303bd24f3ec2e9ba6014ed4b4480c902dbe6ac884a7b4600f0250fe',
                                                                                         'size_bytes': 13480},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/whole_before_after.json': {'sha256': '14cd7e2daa045f92544d1958658adb53433e3df8a07addc4d872951fe04c6114',
                                                                                                       'size_bytes': 99431},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/source_copy_map.json': {'sha256': 'ea43e0ce92e555c4dc43fee7ba4b40c6a3f36bc238b62c02472a11e56552a788',
                                                                                             'size_bytes': 127218},
 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/typed_projection_witness.json': {'sha256': '621849bf861657509ae4a9bdc85e22d4aba4ff7e2e868086d394b26172714136',
                                                                                                      'size_bytes': 101574}}

_ACTUAL_ROLES = {'actual_export': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/observed_slots.json',
 'actual_manifest': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/actual_source_manifest.json',
 'actual_review': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/review/verification.json',
 'authority.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/authority.json',
 'before_notebook.ipynb': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/before_notebook.ipynb',
 'cleanup_owned_identity.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/cleanup_owned_identity.json',
 'completion.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/completion.json',
 'consumed_claim': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/attempt_claim.json',
 'controller_exit': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_dispatch/actual_controller_exit.json',
 'input_manifest': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/input_manifest.json',
 'namespace_observation.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/namespace_observation.json',
 'owned_kernel_identity.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/owned_kernel_identity.json',
 'partial_notebook.ipynb': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/partial_notebook.ipynb',
 'projection_witness': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/typed_projection_witness.json',
 'q008_transition': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/q008_transition_witness.json',
 'raw_transport.jsonl': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/raw_transport.jsonl',
 'result.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/result.json',
 'runtime_adoption': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_dispatch/root_preexecution_adoption.json',
 'whole_before_code_objects.json': 'docs/mastery/evidence/practice16_execution_publication_20261023_v1/raw/practice16_runtime_preparation_v1/run_once/whole_before_code_objects.json'}

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
        raise RuntimeError('1.6 publication binding pending independently accepted actual execution and review')
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

def preflight_practice16_publication(*, root, identity, master_path):
    named = Path(master_path).name == 'master_1_6.py'
    if identity.get('section') != _IDENTITY['section'] and not named:
        from scripts.practice06_execution_checkpoint import preflight_practice06_publication
        return preflight_practice06_publication(root=root, identity=identity, master_path=master_path)
    _require(_same(identity, _IDENTITY) and named, 'Wrong registered 1.6 build identity')
    return _guard(root)

def restore_practice16_notebooks(**kwargs):
    # Resolve at call time to retain existing preservation monkeypatch boundaries.
    from infrastructure.extension.source_preservation import restore_notebooks
    if kwargs['identity'].get('section') != _IDENTITY['section']:
        from scripts.practice06_execution_checkpoint import restore_practice06_notebooks
        return restore_practice06_notebooks(**kwargs)
    # The registered source-pinned recipe cannot be replaced by a caller override.
    _require(kwargs.get('contracts_path') is None, 'Registered 1.6 forbids contracts_path overrides; pinned canonical recipe required')
    checked = preflight_practice16_publication(root=kwargs['root'], identity=kwargs['identity'], master_path=kwargs['master_path'])
    existing = kwargs.get('existing_notebooks')
    _require(type(existing) is dict and set(existing) == {'exercises', 'solutions'}, 'Missing complete existing notebook set')
    supplied = existing['solutions']
    if supplied is None:
        # Existing source-parity tests convert a validated copied master into an
        # empty temporary chapter. The canonical physical after is still required.
        canonical_master = Path(kwargs['root']) / 'infrastructure/chapters/chapter1_transformer_interp/master_1_6.py'
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

def recover_practice16_checkpoint_bytes(root, relative_path, actual_bytes):
    _require(type(relative_path) is str and relative_path == _TARGET and type(actual_bytes) is bytes, 'Unknown checkpoint path or input type')
    checked = _guard(root)
    _require(actual_bytes == checked['after_bytes'], 'Supplied bytes differ from checked physical after bytes')
    return checked['before_bytes']

def load_practice16_checkpoint_json(root, path):
    path = Path(path)
    if path.absolute() != Path(root) / _TARGET:
        from scripts.practice06_execution_checkpoint import load_practice06_checkpoint_json
        return load_practice06_checkpoint_json(root, path)
    actual = _read(root, _TARGET)
    return _decode(recover_practice16_checkpoint_bytes(root, _TARGET, actual))

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
    _require(review['decision'] == 'GO_OBSERVATIONS_ONLY' and review['publication_authorized'] is False and review['cells'] == _SHAPE[1] and review['actual_source_manifest_sha256'] == hashlib.sha256(raw('actual_manifest')).hexdigest(), 'Actual independent review identity changed')
    _require(raw('before_notebook.ipynb') == checked['before_bytes'], 'Actual source book does not equal exact checkpoint')
    runtime_inputs = obj('input_manifest')
    # Physical target necessarily changed; validate the accepted before pin through
    # the exact strict inverse instead of pretending its old raw SHA is current.
    book_pin = runtime_inputs['source_pins'][_TARGET]
    _require(_same(_record(checked['before_bytes']), {'sha256': book_pin['sha256'], 'size_bytes': book_pin['bytes']}), 'Accepted runtime source book pin changed')
    transition = obj('q008_transition')
    relative = 'tests/test_q008_practice_projection.py'
    as_run = runtime_inputs['source_pins'][relative]
    _require(_same(transition['as_run']['record'], {'sha256': as_run['sha256'], 'size_bytes': as_run['bytes']}) and _same(_record(captured[transition['as_run']['copied_to']]), transition['as_run']['record']), 'As-run Q008 source provenance changed')
    _require(transition['prospective']['path'] == relative and _same(_record(_read(root, relative)), transition['prospective']['record']) and _same(_record(captured[transition['prospective']['copy']]), transition['prospective']['record']), 'Prospective Q008 dependency changed')
    adopted = _decode(captured[transition['canonical_06_receipt']])
    _require(adopted['status'] == 'APPLIED_EXACT_121' and adopted['pending'] is None and adopted['plan_sha256'] == 'a3388ccf2a29546a1f196306713324027f472d872ccd8e3e99a92c429b761f04' and relative in adopted['acknowledged'] and relative in adopted['physical_after_matches'], 'Canonical Q008 transition lacks exact 0.6 adoption receipt')
    before, after = checked['before'], checked['after']
    actual_book = obj('partial_notebook.ipynb')
    _require(len(actual_book['cells']) == _SHAPE[0] and _same({k:v for k,v in actual_book.items() if k != 'cells'}, {k:v for k,v in before.items() if k != 'cells'}), 'Raw notebook metadata/cohort changed')
    indices = [entry[0] for entry in _SIGNATURES]
    for index in set(range(_SHAPE[0])) - set(indices):
        _require(_same(actual_book['cells'][index], before['cells'][index]), 'Raw unrelated cell changed')
    transport = [_decode(line) for line in raw('raw_transport.jsonl').splitlines()]
    _require(len(transport) == review['raw_messages'] == 235 and all(type(row) is dict and set(row) == {'monotonic', 'kind', 'cell', 'value'} for row in transport), 'Raw transport cohort/fields changed')
    _require([row['monotonic'] for row in transport] == sorted(row['monotonic'] for row in transport), 'Raw transport order changed')
    result, completion, exit_record = obj('result.json'), obj('completion.json'), obj('controller_exit')
    _require(result['status'] == 'FINAL_RECEIPT_PENDING_COMMIT' and completion['status'] == 'FULL42_CPU_REFERENCE_OBSERVED' and exit_record['source'] == 'exec_command/write_stdin session75951 returnedexit0', 'Wrong actual result/completion/exit state')
    _require(result['cohort_indices'] == result['attempted_indices'] == indices and result['unattempted_indices'] == [] and len(result['observations']) == len(indices), 'Partial actual result cohort')
    _require(completion['result_sha256'] == hashlib.sha256(raw('result.json')).hexdigest() and exit_record['completion_sha256'] == hashlib.sha256(raw('completion.json')).hexdigest(), 'Actual completion/exit linkage differs')
    _require(type(exit_record['returncode']) is int and exit_record['returncode'] == 0 and completion['requires_controller_exit_code'] == 0 and exit_record['retry'] is False, 'Actual controller did not complete cleanly')
    for record in (result, completion):
        _require(all(record[key] is False for key in ('retry', 'scientific_acceptance', 'human_acceptance', 'publication_authorized')), 'Actual scope/claim changed')
    adoption, claim, authority = obj('runtime_adoption'), obj('consumed_claim'), obj('authority.json')
    _require(adoption['status'] == 'ADOPT_ONE_FINITE_RUNTIME_ATTEMPT' and adoption['retry'] is False and adoption['publication_authorized'] is False and claim['status'] == 'CLAIMED_UNKNOWN_CONSUMED' and claim['retry'] is False, 'Missing exclusive consumed runtime authority')
    _require(_same(claim['authority'], authority) and _same(claim['budget'], adoption['budget']) and claim['budget']['max_kernels'] == 1 and authority['adoption_sha256'] == hashlib.sha256(raw('runtime_adoption')).hexdigest(), 'Actual adoption/claim correlation differs')
    _require(receipt['claim']['claim_id'] == hashlib.sha256(raw('consumed_claim')).hexdigest(), 'Derived claim differs from actual consumed claim')
    identity = obj('owned_kernel_identity.json')
    _require(_same(identity, obj('cleanup_owned_identity.json')) and receipt['cleanup']['kernel_id'] == identity['kernel_id'], 'Actual owned kernel identity differs')
    cleanup = result['cleanup']
    _require(cleanup['normal_cleanup'] is True and cleanup['reaped'] is True and type(cleanup['returncode']) is int and cleanup['returncode'] == 0 and cleanup['errors'] == [], 'Actual cleanup unsuccessful')
    started = [row for row in transport if row['kind'] == 'KERNEL_STARTED']
    reaped = [row for row in transport if row['kind'] == 'OWNED_KERNEL_REAPED']
    shutdown = [row for row in transport if row['kind'] == 'SHUTDOWN_REPLY']
    _require(len(started) == len(reaped) == len(shutdown) == 1 and _same(started[0]['value'], identity) and _same(reaped[0]['value'], {'pid': identity['pid'], 'returncode': 0}) and _same(shutdown[0]['value'], cleanup['shutdown_reply']), 'Actual shutdown/reaping transport differs')
    export = obj('actual_export')['slots']
    witness = obj('projection_witness')['cells']
    _require(len(export) == len(witness) == len(observations['rows']) == len(indices), 'Actual export/projection cohort changed')
    for ordinal, (index, observed, mapped, projected) in enumerate(zip(indices, observations['rows'], export, witness, strict=True), 1):
        tag = f'cell_{index:03}'
        # Paths are already a fixed source-pinned finite map, not caller inputs.
        run_prefix = _ACTUAL_ROLES['partial_notebook.ipynb'].rsplit('/', 1)[0]
        attempt_bytes = captured[run_prefix + '/' + tag + '.attempt.json']
        cell_bytes = captured[run_prefix + '/' + tag + '.after_at_reply.json']
        attempt, cell = _decode(attempt_bytes), _decode(cell_bytes)
        _require(attempt['evidence_class'] == 'REAL_JUPYTER_FULL42' and _same(cell, attempt['after_cell']) and _same(cell, actual_book['cells'][index]) and _same(cell, observed['raw_after_cell']), 'Actual raw whole-cell provenance differs')
        _require(mapped['index'] == projected['index'] == observed['index'] == index and mapped['id'] == projected['id'] == cell['id'] and mapped['execution_count'] == projected['execution_count'] == cell['execution_count'] == ordinal, 'Actual export/count position differs')
        _require(mapped['whole_before_sha256'] == digest_object(before['cells'][index]) and mapped['whole_after_sha256'] == digest_object(cell) and mapped['outputs_sha256'] == digest_object(cell['outputs']) and _same(mapped['outputs'], cell['outputs']), 'Actual observed whole-object/slot export differs')
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

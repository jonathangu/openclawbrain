# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30e71eada7154db771ab0903504bad5b650bf53c07abaff3dbef886b8f9ed0b8`
- fixture hash: `sha256-64c4e7be68886fd98adef198061f1396410e569f1ac383e3a1d8328f35e849ab`
- score hash: `sha256-b218ade7730de5403aee2ac2ecb07f9d8d9ab07ea49a47e2261ed537fe3307bf`
- bundle hash: `sha256-8753107cd941c9cbe93085fc7491f1c47838c86a039e83b22f9ccc5bb230677f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-11abf4e14f8e2df5f3f8d1c731716ff1b91d254865900f32b5488290ffdb74a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e6d0cf9972bd0cd24da8d8b646680e75605c2e7dfbb499108e5c287c7653de9b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ab7d54cee19d7d48e6ef603e795e9b73a51b762ec88321ad915c65165477d0dc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7b2f57301c4142abb0d9072452eb3043e2ff1d6312015fe3563ace570800bd3d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc0067bd | sha256-f1f20103cc564d5909f539c007809536e157bad3cbe9f1978dd21e3ff870ca85 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc0067bd | sha256-c7d2223de314615c3c118ddd13b06dcdd88ae73670ae24add3ccbb09e93a157b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-77dd073c | sha256-a25914c0483e879a4542054ffbc7fa251a8275fd6d0cc5bee8915ad2591db4a3 |

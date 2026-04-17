# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0b1bca8dd8d311ca0f474a7d9deb1193514002f9ff0a549efbdfe8a579f7a8a7`
- fixture hash: `sha256-693be8683846991e932bfa4a0d12773f4fe199b9445b669c78493c22255f8959`
- score hash: `sha256-ecb5a371c5726a54a067c0485155f8ce4ccae3b636866dcc3f22785a6e8bf4b5`
- bundle hash: `sha256-c4aa6f2055b1ddf526a3838c896011dfe6734b5139af7c05f6ba69388c68b54e`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a936bbb2a6bddfe389caa1010c92a0418532436fd2f50651530e961a6495d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8efbe5a0d399fdb0b4d18cfa3e92682d617af59c88497f32952b6309c54b8dea |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-def29f352bcd82a178277def926ae7fdb80998fb5066744a16726d839b4e5045 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b0d075dcc1e5f6176783ebf86669788478b7899b19f73b0a373130ece7f65ff1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d2887391 | sha256-9e8d8b3f1fe165f3909b7880b5facebe316f83afc12f8504ab1b8ab88067efa9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d2887391 | sha256-aaf6ec3dd47f8f99fe632a3b46a9ed2b27969eb0bd5c67d6f693b3186acf8533 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-640a7c5a | sha256-e54429de4cdf5b4e086e9b4ad420fb2c08d3ac903bfc5f47321e29e99d9d29cd |

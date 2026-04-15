# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69683abc74146be49e8afbd73d2f629322351b8f1ff326bedad7089f23b35cc`
- fixture hash: `sha256-78ae89352ee0e2620fdc9e4b5d6b74ee70bb4cf28775ccac9315ef7f4b6b2525`
- score hash: `sha256-42eb5a948031099d5a7a93f331f827752e63ff8c08ea58542c3f8e28709603ed`
- bundle hash: `sha256-c5c40344adebf4bd7ad4c1dc30cf1e67bd211e24fc6331e6bcfcc0bdaaa5d908`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81a98d75515ca1c6519d32d4f8b5120338f9765022c93b90e0504e9561ef38af |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aae3ed46c9f4a4a41ba4899bed3cc731c929e71bd0605cfe16827cf3bcadfd92 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1322804881f222b0c6ac1067c19eea2fd18e7b99c389b6f6418bf74b012f936a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-367ef68297f12955578ff7d68c66974b10a77cd21ec6d94bd1cf1263f2bdabaf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a23d053 | sha256-4e3cd01b4698a6e0730839548e945825abd54e16c3c6191a2268c5799249184c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a23d053 | sha256-5770dc1c4b95b8e2061284086619188841cbc7f6ee8e306212acff954cfb1c35 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3a23d053 | sha256-4e3cd01b4698a6e0730839548e945825abd54e16c3c6191a2268c5799249184c |

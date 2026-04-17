# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bda6b3da4ef39b29be45310328eb0566a39d316769663e62675a6105dd7880f7`
- fixture hash: `sha256-e12b530a582d1487040cb7cdaf3e1255576e9298c334dbf79363d1f81080b1c8`
- score hash: `sha256-0083a6534a9fa841c557cc215d02c114edf64684f68b3c0b986cc2fc681fc498`
- bundle hash: `sha256-3041272027e79b3e927cfc4ea6d70b2cc441d1649380c85e92fb3c7d86b344a1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bab067ff9232cc412579013f9b35dc498686eb53e7f83b8de58e12e80ba3c742 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-274a9f84e65bcdd7238f534008c94f61d03c8a90dffdfec5075e308e831b15d0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7ba61a716f2a136f07638bcb762a924c48948fdb5a9e1906f49fcb8e8e3245c6 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-cb50ae943dff3afd808f35359ece34cbb3d902136ba86ef4fc45e040aef5ab5f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-edc0ebe6 | sha256-6e4ac0868778816606875fa0b05d7ef4ba98ad435362029c81c023a6edf0063e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-edc0ebe6 | sha256-fb234aba6dfaa697cc9843eef306388be0b2bbda14156bf20393a574a6ea6a7d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dbd34105 | sha256-c142110deef15843178347deb1f5a4af2cd6876e2664d1cb4f782892aeb68a1f |

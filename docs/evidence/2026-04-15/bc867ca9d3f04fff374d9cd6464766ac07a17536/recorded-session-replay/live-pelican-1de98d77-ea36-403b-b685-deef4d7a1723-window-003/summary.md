# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-db3086ba9270f5f47434d4a3c708e73ff2624adb056b71992e75ebf839a91592`
- fixture hash: `sha256-2a8d321cab2bd435ac998d63d68b17b7fce95e9a0ea6d02ef75e09676d4240bd`
- score hash: `sha256-1b420ab1c732986f21ed41acd0ae626164bc49bb4a0caa0622d6465fe8eb4a66`
- bundle hash: `sha256-898d443466868037e5f96e7a91ea384d3dee78afa4ea5d7ef743fbaa0a97ceb1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da6633515ce74e28e9f8bbc2cc587b6b0548deffeab6470c77c67fc675828106 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ba53438ad40f0d4ddcbbeb6d5c26e644abf97b9b9d64c284d5682900a4392e69 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-202e59be97b5156f01dc626479260eb2531bbbb2f0e1246833555f59b793d1a0 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1c459d92995d0544ada1bcfe9f01239c8db11369cb88e7443de63066a9c242a6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-eea41849 | sha256-f4090fbf94555303e9f4138d87bbf5c0359d128fd000512d2640ce7878f712f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-eea41849 | sha256-1d7f7a7421a09a8dd94b57b7354dab4dbfb595eafb79e8464d1467697672f2e5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-eea41849 | sha256-f4090fbf94555303e9f4138d87bbf5c0359d128fd000512d2640ce7878f712f7 |

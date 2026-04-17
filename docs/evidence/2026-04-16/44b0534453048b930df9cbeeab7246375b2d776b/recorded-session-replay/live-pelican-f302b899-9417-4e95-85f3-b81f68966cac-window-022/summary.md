# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5d4b7ac6ed69712b1588ada5d64482dda6216ae5bbb670a70c4e5011448ae050`
- fixture hash: `sha256-c583a0a30dc7272198329e0ce06b64ff4fe39dce1f96b56a4f82e04f4a924ee7`
- score hash: `sha256-6dbe3500bb74f23874b45037e805a0389b966ad61089cb3c06aa668aafae0a74`
- bundle hash: `sha256-0af14538fdfed2f33bb19c227f7ee443d547e607bc2e7dd7ebadab35cc00f735`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-646908dce1c2aa715ec563720c445a9dc7233e215511f30956abcb8a6c0f9113 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c4172881ebc73defeb3ef63d70ff1bcc48d177765e9439f9140b3ce260f74047 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea17e6d7d569c672654dd8b2c49e734271c5ad3d00f234e347e614d3b8c134d7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f1d1f94b166c394c59c48862538c9fb400c13c9afa520bc6caee0cfb5cbaf893 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ea2082d6 | sha256-367e29202412e56a2d29e138e8f03a11d824204eb0f47268ddb9c4fb361b4531 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ea2082d6 | sha256-5f34f2f94bee1aa7c77a727175c6ee674f50aed418bf87bf4e0d2a844639a206 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d78da559 | sha256-cd59f735ce217db2cf5a144a24f87711e4e1343f58f6327064d49af71c9a0e10 |

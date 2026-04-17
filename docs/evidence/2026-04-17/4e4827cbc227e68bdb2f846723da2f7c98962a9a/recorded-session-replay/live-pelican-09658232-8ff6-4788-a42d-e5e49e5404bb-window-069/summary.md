# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069`
- winner mode: `graph_prior_only`
- trace hash: `sha256-836584be6983eaf5fa6eb8781cac34bfbc9fd538fa4b161ac2d1263fee14146c`
- fixture hash: `sha256-d2752f3a765e793797ebfa0ab38ae1044dcc8b2c28b548d73dcfade2be50b251`
- score hash: `sha256-22fd390ab8d87cd8620ce70f4ec723c2cd1aecc49282bd345e05293d51603898`
- bundle hash: `sha256-ae85461f83d2cabd0a3b080385358dadfa818512cdb321809924bcdc0e176956`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c391bbc2363181b72ba8549d9009dc5fb197cd45a7341e37f0fa91e51803c6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d432a700a6c50f701e4324d05e19ec6a838edb086a6e8ddfde668df9f84a469c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f10bed4518003c41724a19ac73380f5128e911a515c3e5eed056728e69ce8280 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e7f69ef24127f6b85a3b077c01763d8b5c2f57310217ba2df906dd163276595e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c88b94d7 | sha256-16fe02af2204ec5bc6fccb3a9516d2a27906c769227ce5c57b51506d5432af16 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c88b94d7 | sha256-38818cc23920cabf8da451a5b5d91d6cda50ba39badc0bf23b02f3e95127acdd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1326c678 | sha256-4b47e3d4ce0b9628c63044d34f00b85681567db1c0fde187812486932b6dc7cf |

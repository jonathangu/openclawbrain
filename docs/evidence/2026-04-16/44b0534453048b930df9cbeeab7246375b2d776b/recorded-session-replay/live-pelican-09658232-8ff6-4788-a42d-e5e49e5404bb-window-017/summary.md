# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3ae37b035b14582a9db80eca92cf9ea284e1a083da7e40c15c27766593c501ab`
- fixture hash: `sha256-c1a05a74d2fece7febaade02118c0528463204c7c70c4fc0e050990958f60a91`
- score hash: `sha256-55c402b2728fffc7006555ecf17f8831cc0cbff4076e42e47b58b492d7c1b33b`
- bundle hash: `sha256-5cbeaaa12f463289b084e06f319432486448c514608b7870c60467eacbe80739`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bfccd81fb07da2148e74d03332b298ae8343e32f9c89c9de3c815764af2fb42 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f58981f33e9b031036010dad2ff233de5a81c3c75fb9410c57d8a711952909cf |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb44ae49415169f78f8f03556843adda873b2c537959ec82e570ae4acf928727 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-971ac7f68a0f09888747ed371ab0263bde2de019b493ac0ce990d88ca2648c40 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3b68ad0 | sha256-f92083ac9a8b949a4be58e446319d4fab5748dd23153f65f35ff633cda107fca |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c3b68ad0 | sha256-c7b86647ae84a9b4bf98a10d3af175baa3e74327078ea1e5b5902e94aa0b1d60 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7c11babd | sha256-cd81856dc404e851826bb179279803544192381e56108839d002453ee0cabb6f |

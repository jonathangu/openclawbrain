# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67782e30fe5f9982125f26c2ecd77317f6b86c34b8443a476ff968e4172fc9ad`
- fixture hash: `sha256-3275c723fd5e55770c99a0a3826bd67e0749405b630c9523de493fe0719c674f`
- score hash: `sha256-a296d4f7de890afab891eb71f11bb3c304a06c17cd649304acab1e2275ac7e1b`
- bundle hash: `sha256-00488b99e1aa5a57b470b1f9b17a9776dcc1805fcf70e785e5855de1133d4eec`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6faa741f27c297696cddf75c51e07e62f9d376795b5d33f012fd6c625e199a2d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f64ce6a7b7fe7c18eedf1a903c788288800aee15984c81ec32fb74380a9e849 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b205896380aba8b0c1e24947505f89cabe1df9e508f3a36957f53f83266d134e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-089e3ed81d6d48dfde950566d11d7769ae79770dfb893b17d29ec69f49c44526 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9378666 | sha256-2f976711fa5577fe5e27d7ccc5b735ca67754d3a4007b9dca0fffb5719f00121 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9378666 | sha256-b31193aea09494c9e9f35d623ca2f79132abb79b4a372083883a6b92648c242d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ce66d14d | sha256-463d5ba97d9dbd302a0596209cdd398a42ef82bf64e28cd311ff3511752606d8 |

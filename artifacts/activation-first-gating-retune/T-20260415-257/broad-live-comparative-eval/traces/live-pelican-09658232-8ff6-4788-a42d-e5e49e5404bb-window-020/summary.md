# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f3b511c142861747542dff1ddae4669701bc9656bef363a96e4508cee5f2a20`
- fixture hash: `sha256-2db80cfb229c04864b42f8f3b0cbec60d6dc032d77659032291a70b2cac64512`
- score hash: `sha256-54ebe275a4d8b816a9192286cfa69956673d79e07c7e4012f821505c899eee82`
- bundle hash: `sha256-7334480546a31c37c00d2e8aba4f393a91df28078fd5502519360ec43b56a932`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-03f1fe0b1ca7c86742bc307098d07423af66afa6b8715bd5d40ceee92e59b30f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-627ddc7f582ee0661d064b382539634c22f658c30669743b6759f258e0e58868 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f940eba1c4c4281a3e2cc2701c719140d2e1d9e3066cabcead112f804133be5d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-17ac63b3046948bdc8eed73ffed8bea7b8fe905c7bdff913ec60285825cb9aaa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fbb383d4 | sha256-b084eae7fc5cd6939f87cf327d5719d4ecad07d24e65d106b5d428e7d5715e65 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fbb383d4 | sha256-d56ce0b57d76bae15f0fae1993c1410e0f17281294df36efe27c3bcff966ea64 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-282391e9 | sha256-db142434e9be25e1a99874f251d85fb55c9520a9c1a2a0c74195c046ab81bc5e |

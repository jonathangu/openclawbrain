# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f3b511c142861747542dff1ddae4669701bc9656bef363a96e4508cee5f2a20`
- fixture hash: `sha256-2db80cfb229c04864b42f8f3b0cbec60d6dc032d77659032291a70b2cac64512`
- score hash: `sha256-ff8474487ebfe82fefcd01398069bdaf58c7764d92c4428e6c4745ac7be24016`
- bundle hash: `sha256-10bedf4539104400763dbe1d13341c61ebae86e9c929d29280a4a3cc2d43ada1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-03f1fe0b1ca7c86742bc307098d07423af66afa6b8715bd5d40ceee92e59b30f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd1151aaeb6cbfea35b6c8fb55c4f238a5678c22fce02b7645c54402b676ea97 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e587e28799146fe7d5948896a54abbc489ae9e8127b4f37ded58b5bef57f657 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f4a6714c654d569333840ccad06df8fd40a62b9259a454458f277069ed61a546 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e605cf6b | sha256-cacf4b1c6875cd2148a23107b2061afdb021fce99d04149a07694483ad6eb39e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e605cf6b | sha256-d27ceb7d6974f598b0be49f6a4c1c87b1abd3ff2e469f0f07adba54fb6c5f656 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1275dd80 | sha256-530f42d788501c5a98ea474e8ff1a48e221aa5330f4739ce1e94d550da4fcefc |

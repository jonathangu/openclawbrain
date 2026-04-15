# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f3b511c142861747542dff1ddae4669701bc9656bef363a96e4508cee5f2a20`
- fixture hash: `sha256-2db80cfb229c04864b42f8f3b0cbec60d6dc032d77659032291a70b2cac64512`
- score hash: `sha256-f4765749e9c5798e5dec301acc6313f07a5c7894e3d0abd1e69f462809939785`
- bundle hash: `sha256-75b11962c240446109605c0f6af720e5d27be1561f47ab6b92ddc1ae961f1ddc`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-00744b1cd3d9988b84029ed1bc63535fa472bdcbdb542196079739725d228248 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2dee53d2c5b6e3eb81adc30ae4f47d0302dc6419b426f50263f9f4f6f85d7f1a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2ef79618a8516a202cc1b2e233ebda688c5ea180472d3eab97de7568150cf4ed |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-474ba67f | sha256-1a610779d4f22e423e523ef427ef96f6bf79e6416efa14cb7042b7cb4cc779ac |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-474ba67f | sha256-120ca3c085faa546b6ffdce062e4256bfaed0c8acab5850f8da2b302ff8f0799 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-474ba67f | sha256-1a610779d4f22e423e523ef427ef96f6bf79e6416efa14cb7042b7cb4cc779ac |

# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad8a200767aafe991ee7054e677a19f37758804d5a9a487f59ccad4263c83187`
- fixture hash: `sha256-23ce3445f512fae9ac35202b97a34c12c8d0db3c79197541a8b90358597638a3`
- score hash: `sha256-b55d39d215c38901d5c2c9fbc26cf5b71a6ea8e3490bbb85e18dc8eae03f941b`
- bundle hash: `sha256-54bcf737478ca4728cab9031630d5d15561137fca938fb4d8babce045fcd9049`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b757ba8fadc84f09e0e7aed31f0b4ebd54fa8fe354fc559aafe046aa0541083 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9a35887892a8915707713c70c321e510dde45de8dfea88b10d4ca7e93840a2d8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7bd6b6121260461c82c912914d8e7b75dd5e552e14835cdf1cf01b999df9d8ca |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-75ccc3c0d44fffd9794ce8a301b17b27f5780c178d9d43b5b3b27995841b0cc2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b64c9c9d | sha256-8a463e36fd5ccea69959bc9ca1087782032b69a5901abbebe7845822ae87916d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b64c9c9d | sha256-297e6b6b74f59cb1acfbf11d4add493011e3486496a32c02bd41893c02f1935d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-4111933a | sha256-1b4edaf02441ac221d260b940f88aec5c17e7ad70dbab13ccafd6b75016c34a6 |

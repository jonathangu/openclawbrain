# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b834ef975b4127c9fa6cce4b12fc80a10ac45f8451c226003f814696763d3404`
- fixture hash: `sha256-2ef4285ae644d199abac210f4e94c99bfd3cbffd40a56868154cea15ccdb9a86`
- score hash: `sha256-9d8a006807e823d94dd557b9139851f304b610fc755e37614ca94b135a26f3a3`
- bundle hash: `sha256-360a892700a5bb875aa3e0c03d54f8abd3c55038f613e26c15900d8cdac4a044`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f63fdc694dc12e3fd1585f1a9d1a8d63286b83507ef0e36210c868d071e50d26 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31687f0ffb5e4f9affc359c330e1370fe167b2cd07ff8cbdd0f7c4cbca729ef5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7afb08815001b85412cbb1c5625765b9d0eb481cc45170d2c251d6503562f734 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-035ed75791b72befbb2757dd4877359febc2c2dcebeca287d27b7566fc924e36 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f94ebb16 | sha256-f9e904c06621614d4a5122850d6032a6f434e4b051cb7d0a7579463f8eaf26a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f94ebb16 | sha256-a4abe2e6653ecbad86d5dae33deca271bb899a57683a3904810cc4da1bcf36b3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f94ebb16 | sha256-f9e904c06621614d4a5122850d6032a6f434e4b051cb7d0a7579463f8eaf26a3 |

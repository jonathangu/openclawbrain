# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4ecb7ad01ebf51dd3aeb7754e784e70eea9f4067a9392ad81778aff88de83b03`
- fixture hash: `sha256-ad7aadbe694390cc07af980435b05bd2086d5294c79bda5f4f75ff348a4a3b75`
- score hash: `sha256-65ad3e6fe16f30bf289a24c0b2778d17197d59b7795254a8d1df86ada8bc727a`
- bundle hash: `sha256-54d2245b8fae990ea93f7dda8656edb56972962ca7a55ca7c51a99a123382c5b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8a8a0e6d7dd1a7545143681fd0202299acfcab2ad5ce85ed5e5cddd516c7f67 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b0f3e1672e1ad285c2ff574ed320720f819b6d73ab9528ae68bc70306b4a01b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09589cd0a02c1561abcb60819555260324c8faae483cab7f33eccf02f6459c67 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d1616043e0dbd28c5e53c4fb711bfb546cf86349f0dbf87b8d6ca7653a3eb9a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-75048f11 | sha256-c2f0cfe8ba186569a033f3a0aea68a11cf480890a1519e52191cea21388eb3c0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-75048f11 | sha256-8e3c26bfe7b56658ad5b13a62459a1f7e33247781d5977d25dc3488b8451fa22 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-14f6ae44 | sha256-516490cace4049657e90988b22632f545df92c4e994069a99c529510c332e427 |

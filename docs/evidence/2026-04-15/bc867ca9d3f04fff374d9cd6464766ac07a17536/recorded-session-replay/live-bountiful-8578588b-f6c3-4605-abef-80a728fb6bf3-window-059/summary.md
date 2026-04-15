# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f67214506d27201aa84ea9f3606a2dd530633ac1dadc31f05ed3e5af4326650`
- fixture hash: `sha256-a8505bcc6501d6f6856db49c4b5c901ea18f4f407cc8520314fe72513fbff478`
- score hash: `sha256-97ef10669676f66540778e95cf2ce7fcbee36d69052d6c5e74934f03e8d08002`
- bundle hash: `sha256-e5499f8d8ee6ad85e6529f111e97462c7d607deec99a4674052a226619087cdb`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-44d92ff0a09445bae66b58b53a8f96e059e7981c4a7d5440523a3b87ed99e3f4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-95f2aa435e509cd1f8e9ffc3c0b0f286c27eddb47b5a2bf5ad9d3c58d6df2ef6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b37edb997c54f86eafaa18eb0c342fe91cf543607cffcebb65869c928b462f76 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9dc95b66831e0ee4d3fca6b60de7c33412583f07ea90ec1363172bddb9790b77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64ee2e06 | sha256-731c901e7279ca8dedbac720861bb519265168fe55a4f2898981e71a256e48fb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-64ee2e06 | sha256-872f898f9fe97d9a7346dc1f6ace8eb82946d1cad3950d591aac2fa10ec8bb13 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-64ee2e06 | sha256-731c901e7279ca8dedbac720861bb519265168fe55a4f2898981e71a256e48fb |

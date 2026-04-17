# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1d9740289fc2adbace7590e78dff24d1d94c6a419d6474e3af27754996da05a`
- fixture hash: `sha256-b9420b72d3a2c2c9c62adbc0b7f3ef24407bf200cf73b9c382cce44e2d33fe6a`
- score hash: `sha256-0d5665c5f96dd4f70d955ddbf89583907fcec3714a58ab5fb1fc0112cb3ac022`
- bundle hash: `sha256-5cda49d93182274659006f235a52afd9c67828f6d393eedefde74074fca3ae2e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36b1da37ad1b5e7a8a6bf9a89b082e6da9affb9cefe62c4630aaf0bc52cbd76 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e5f9ce7eb07d8711f66e85722d4c7938b82b19feb0342a1de304585b7bdec3fd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9fbcfb6fb06069b4b2d93225f0a8015dbed852adcc5dab2c0c8ece44a2a0985e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c6f4b64327d0f6513d9ccab4d774a4423b532e3cd7f673974af5278745b35046 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-60b1ba1a | sha256-92a8f00bd3e320c179dcdb9358e63e0faa8c6db731c986db81650881496e3dc1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-60b1ba1a | sha256-629142c8e3cbeb104ca8a536002fcec6aa0a9abc49745a5130b8b3be13df0f9e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-502d5581 | sha256-b530e496019e835dcf12eb307db5ca387edda61330787a237393cb6262d10483 |

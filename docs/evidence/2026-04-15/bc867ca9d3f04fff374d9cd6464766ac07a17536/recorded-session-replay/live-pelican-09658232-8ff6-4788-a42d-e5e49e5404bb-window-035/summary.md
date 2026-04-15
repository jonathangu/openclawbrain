# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-1748625a14b7ac2b74478c764701b04d5290840fd9b65ef8e114c2189f9a67ad`
- bundle hash: `sha256-e18137c972a39232aaef3886e5a319cb4e84639ea3b8779cc552a505b84073f3`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1623aa98a961961a182098cbb09dfbf96da5584b9efee0863f57cb38d7ebe41e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-6577f6aa7f4668b91c5f6738a5fecce3121953c30918e80948092132ad778787 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe1e3cf83af2a22c2bbca4f4faf6acbb98f096424bc10c9a654fb3a04cba55a9 |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-a316831ead43e7890d0be422f4c77a6836f938358a096db6acc112682e2da823 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-1959d3dd | sha256-15e9a9c13df0d8a5b16cd38f19a34be43ca5733cc89a418c4e70b39dbdc3556e |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-1959d3dd | sha256-2a1623ac51258045de79cc8bd2f52333b83a8d9d42d397efb0bd5484a66f8ba2 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-1959d3dd | sha256-15e9a9c13df0d8a5b16cd38f19a34be43ca5733cc89a418c4e70b39dbdc3556e |

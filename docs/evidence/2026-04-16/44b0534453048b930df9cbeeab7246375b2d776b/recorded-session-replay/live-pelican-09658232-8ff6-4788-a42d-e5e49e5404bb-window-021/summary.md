# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba20557b6de7502d32e3b83fc90ffd6f8ac19ac17c3a2b682f0895bf8bb69c7c`
- fixture hash: `sha256-024c24fcc3f69f4d62a086b795f3d8c9e3625b36454d26d1e235e1664a651060`
- score hash: `sha256-2b314b9369cf551486704783d79c46fd7a11393827b0daf7b7f70677822b55e1`
- bundle hash: `sha256-b2ba89e05121cb611bff69b72ddd1719e041c3a94f8f43454a5f4f3335c0a80f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91ace2bc1ee23370cade1ee9720612db55ace83c09692395cb273562a40c2beb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ea11824c529e92d6901fc6b1db59dc08bf420ffb5c471118dda37a29daa95f3d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-af05c1c089d91faee282393001a750162e5d917dc8c1a97362a36a80f291f6b8 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-5363eaf46a948e5c5c9098ba89c105cd9aa731c8044be1b3caa628565b610fb1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-94aff36e | sha256-18a79d65870c1039fff71edfa4f4388e09a4851ef5d50fa06483a668534af2e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-94aff36e | sha256-3eaf9066ce48e2aa1a10093c21450202bee7e1e104f9781146f58c939778ec6c |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0302cd97 | sha256-f172c8c6ce37cbddc8d5a6c7b48a7cec693776a0517e83bec2cedfb38d8d8e9a |

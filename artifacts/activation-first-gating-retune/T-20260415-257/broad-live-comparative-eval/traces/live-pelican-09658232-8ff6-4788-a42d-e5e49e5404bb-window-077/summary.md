# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb0ffa9f4e8d70c4fc8ddccf35cb362423daf7116804c6734e40b9d0f4296bf`
- fixture hash: `sha256-810d932d8ac4f8eed98f074f82298ad7f5b0354d5fdf19533c533df6c21240d2`
- score hash: `sha256-7cf293d4e2e707907e9a4ac013f48076b1cfc219b9339963a8e91765ac958945`
- bundle hash: `sha256-944842c45b3fcc532c55aee04e059492aa0592eb04658347bec48b56f7169f29`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe2cee15d245f859cb5315bfc802316abb7874a3bff97839e84f3440b5d4a896 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6ac194845cb6c9c1cda1e55ec6c548711978a1a7a267ac643cc0ab70c8418cf |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aead9bff773cb2f1e4e4f9bb902474bb87e58d6d88a6aba33b26c9d4e65bc8c2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b48fb5f2c6f506db76c5ae5e0c47e8f9dab049ffcef09187f7bc2d03e83a9a55 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a7a2d2a | sha256-a84188969110b63e4b900cefd0fa3120d8174b2757c3cca4faa6b6e413dccb79 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a7a2d2a | sha256-75b6f32018f82aef6467bf2b34611f32a2fbcb16b633fc401402d73c1111ee05 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-af6a3677 | sha256-2865f7982ea8e1b8b00c604b45386747bd308f7d108610b96d4fbde8283b098b |

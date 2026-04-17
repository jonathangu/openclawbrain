# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3188dea0835fbe3a5c0a4bde0dceb823e483b3aad858e66d490cbabd38ee5d72`
- fixture hash: `sha256-c551d2cb8b10201e8079d270837622ef96e1675624dead00decec0e3fb02a4b9`
- score hash: `sha256-e6b04b5eff64d391d9592b61414612c26c004bd009bd5c062e68cc2812e0bcc7`
- bundle hash: `sha256-d34f2465b14a92191c7feaa872db8bd75cf54f6efb5fba09ab8786f28168fe70`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7e263c626ee395c7a005cdac6d8c14b4d8e92d0d3065cdc0b98a11e431231d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c642196ee726165fa1282129e44e611b4f722b08f75c2f0863c56f1d339a2d7c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6da5aa0b9db1b44e52ef0a7a2a03d5094b85c7131132342ad2a1e9b5dcf2ca85 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-68b1a70478f5a9ba147db5d9d17544914181e9870f184ab1652e926872b58da1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bbeed734 | sha256-46cdff3b5967b80870a3d945523de71935a230acec733857e8c30abb056d820d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bbeed734 | sha256-6bc4a14afddecc4af43c1edf7c6a33bc12608598638156655ed1a4931a1b66fe |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f7b494d9 | sha256-c69c38c4a3423c8f8776b6f1e3e0aad0dde58920268b9e7903f30fb2d1b64772 |

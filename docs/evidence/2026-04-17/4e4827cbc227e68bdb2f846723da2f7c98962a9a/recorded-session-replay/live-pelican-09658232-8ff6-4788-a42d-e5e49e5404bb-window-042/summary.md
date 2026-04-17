# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8470650c25739a12e09f620484c19ed76e535bf12ac60fa5bc19c4d9e71da263`
- fixture hash: `sha256-1a290f6c39ed84b2ca073e21a57823e82667ab7d1408676870645010e286d76d`
- score hash: `sha256-b7e3e1aa83be0969e32f0d8dad091af5612529e434b372ee480290d1ce3fa245`
- bundle hash: `sha256-16ac6062756ef6d9724c063f69a0d6f60a0240eb23cd1be432faae08531b78eb`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-575c5da35d27014ccbfd8fb043d25f84e1287b134d1db92502cb2f005c370afe |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8cfd06d43efe7553d0f72eb4392522b634727354e649798c5011e2661a41a45e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-151be711c46e77d7b296d8e83bcc83f12d60d78853cbaf9d4b0ba2effb9e0752 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-eefae0231a19e7de69c2a2f41f48832cfa7d294221731bd9f0e36062fd62c86a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-acb9b0bb | sha256-17bf08b6623826c67e8eccceab5c4776ca32e1c98a0d237b3bc21fea9cb7c565 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-acb9b0bb | sha256-e9382781db8a095cb366397eaa87502a46b49744b0146dd9fbfcf8743673e8f2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-77479f08 | sha256-7adcbc12ec7de45a6bee07f30caeeab5b4fcdd88c07e1b6482c82a491b2dbc53 |

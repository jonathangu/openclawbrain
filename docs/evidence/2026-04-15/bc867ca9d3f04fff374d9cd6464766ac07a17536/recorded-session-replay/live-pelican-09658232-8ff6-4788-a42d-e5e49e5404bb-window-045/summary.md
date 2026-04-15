# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42dd5ae1fbc52ab37ab26b7eff707ccc814072dbaeb4cf80246f57beb5474c7c`
- fixture hash: `sha256-dae1773b38ede59c62f735546227926063dcc22433a680794834acb15197b82c`
- score hash: `sha256-372fcff966051070bd0c511386c6ecef9ef258d82ae2ecac20a6fc1c6e6c7a8a`
- bundle hash: `sha256-dd707ca19ef0b4da80aac92f5386e6801a94eaf3032b17dc802c62e505518537`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97339b57eb3bff564bd492b91102981f9054b332cee78d9338b804ad8b646434 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-777b68bf9e151689f6b36f599e9935cccc4cd55e873c5846614ff55e1ce55e67 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-51695d9e30da439552186be35e7773a233db5a14b1e15368637410ca20ff597d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2a19f78ab07d18b08078dec157c4be3f68f676647903eca9e771857b6f3c3307 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-223915af | sha256-979bf9f03c4a1d6dac2cca3a935ff52305c2e858fd02af9bf594c4e6e1fbaffb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-223915af | sha256-eae4a22f4f5a2e683a1a7292ad908b049d8d4ecb777c355464bb2e49e66ee666 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-223915af | sha256-979bf9f03c4a1d6dac2cca3a935ff52305c2e858fd02af9bf594c4e6e1fbaffb |

# Recorded Session Replay Proof Bundle

- trace id: `live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-93228e668a08c975492dc6af4e3bb4c71052274e3e003bc535d1e798cb5b7551`
- fixture hash: `sha256-de7894b208900137452009cdc652956a77d2f2658869966be1c1f8a47a12873b`
- score hash: `sha256-79ee13f9e41627b633653cfc6c678d0d07e26ccf4331e45315caeac78dd05eda`
- bundle hash: `sha256-9143e3ef423b46a681a1b94f9b88d7a0250aa0dfcff9db17514d6e1f9b3dc3dc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c56ee3ae997453b8eb93280de0f46e35ef0156aa279e2ba51ceb2f8a8bfd749a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-abb7456dc94457dad6a75655a03f2412c8ea315daafcbb002eec82fe2ef84f37 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ce09c8a8aa89c8ba8ff16449a624fb1ad7073174f31870aaec4492319dbbbef4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3cf6c1a0f293a7a104982acd465d7fbf65b10a0ff93660096a9834c941d99860 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2fe992cf | sha256-46fd35e975f5fa8d37fcce8b97556ff17e1859fc4419d3fac86728db3316662f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2fe992cf | sha256-6364591c98ead3f66db751be6058a8fd7d0d1bf85eb624293913decc276c5638 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-29df6808 | sha256-a84444f84b06414736f34fa9fdec167b34fc2be7b1f2594664eab11b43f62cae |

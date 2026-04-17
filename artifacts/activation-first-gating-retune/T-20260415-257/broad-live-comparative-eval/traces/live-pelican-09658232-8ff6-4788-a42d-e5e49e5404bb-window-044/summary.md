# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0ad01cba10197b8b05f27f973f11a9d5ae36ed08ae5a28bf29003ec749435fdb`
- fixture hash: `sha256-6830cd222cfba386b49de3a4d46620d84d8c333ee746eafd9c6c6f8ee2dfa95c`
- score hash: `sha256-cc6c04df23ad23b96cd2e293ed4a441ad5cf724a27459e796564d0d3449bc634`
- bundle hash: `sha256-bcfb569f4db5c69dbec5375ff88831b2da7914aa199d2513224109556cdf67d7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-490c3e2fb439b8787ca9fe3cb573a9c587cc3abe414e2faf637ea2b84d91d268 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cf18da30c2e3294809330cb6dd52db8a6bc9169ab6b65faeaf4e2103b35e39eb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-db2a9e3e2766a5cade409ae3852f669d26996f961e877bde7605b9677f1c248c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3659ef5ce233f8412a3fae26ff5ee8e0ec5da6e343193ddcf53f26a501bf2f7d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66e23c15 | sha256-09580c966a9bc44dda256bd77401c485ade680a07c8cabe66c00b0f36d787aae |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66e23c15 | sha256-c8190f904749b3910bfb9a17c3309a74f1308f89cd4f83b1b89514e650eb6392 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8772dfde | sha256-49ef1dd602ead6939861bb85e0bc4ecaf258a73ab9c7094d81e3f67725cc679b |

Given this CSV data of headlines, give me a "tone" score depending on the headline text. The tone score should be from [-1, 1], with -1 being extremely negative and +1 being extremely positive. 0.0 should represent a neutral toned headline. It should be roughly a Gaussian, with most scores between -0.4 and 0.4. Output a json array of objects each with a "date", "headline", and "label" parameter.

```csv
<csv data>
```
# AAA-Explainability

Explainability with SHAP and LIME

## Structure

### Deliveries

Our deliveries to the course professor. Includes report, presentations and any other necessary artifact.

### Papers

Our reference papers to explain what is SHAP and LIME. Since it's not given in the course, we took some time to understand some of the ins and outs of these tools.

### Source

The code developed. It's divided in 4 parts, and each has the goals it tries to achieve.

### Environment

The `conda.yaml` is the file with all the requirements we needed to make this investigation.

#### Create 

To create this environment do the following command:  

```bash
conda env create -f conda.yaml
```

#### Change to project environment

```bash
conda activate aaa-project
```

#### Update

[(SO Source)](https://stackoverflow.com/questions/58272405/how-to-install-packages-from-yaml-file-in-conda)  
To update this environment with new packages/versions, do the following command:  

(Without the environment active)


```bash
conda env update -n aaa-project --file conda.yaml
```

(With the environment active)


```bash
conda env update --file conda.yaml
```
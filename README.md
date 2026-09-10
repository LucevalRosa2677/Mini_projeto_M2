# Inspeção Visual de Peças de Fundição Metálica

Mini-projeto de **visão computacional** e **aprendizado profundo** para classificação binária de peças fundidas (**OK** vs **defeituosa**), com o dataset *Casting Product Image Data for Quality Inspection*.

O fluxo reproduz uma rotina industrial em seis sprints: ingestão dos dados, diagnóstico clássico com OpenCV, morfologia para isolar falhas, pipeline `tf.data` com augmentation, CNN sequencial e auditoria das curvas de treino.

## Estrutura do repositório

```text
Mini_projeto_M2/
├── casting_data/
│   ├── def_front/          # peças com defeito
│   └── ok_front/           # peças aprovadas
├── inspecao_visual_fundicao.ipynb
├── requirements.txt
├── setup_git_history.sh    # recria histórico de commits por sprint
└── README.md
```

O diretório `casting_data/` **não** é versionado (imagens pesadas). Use o dataset local ou o download automático via Google Drive no notebook.

## Ambiente

- Python 3.10+ recomendado  
- TensorFlow 2.15+ (API `tf.keras`)  
- OpenCV, NumPy, Matplotlib, gdown, Jupyter  

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook inspecao_visual_fundicao.ipynb
```

## Dataset

- Fonte: Casting Product Image Data for Quality Inspection  
- Drive (fallback do notebook): `https://drive.google.com/file/d/1NZOjCHDRrpn7PmbFKVqegUP5arfdXHKK/view?usp=sharing`  
- Layout esperado:

```text
casting_data/
  def_front/*.jpeg
  ok_front/*.jpeg
```

Se `casting_data/` já existir com as duas pastas, o notebook **não** baixa de novo.

## Sprints

| Sprint | Conteúdo |
|--------|----------|
| 1 | Seeds (`42`), ingestão local ou Drive |
| 2 | Cinza, `GaussianBlur` vs `medianBlur` |
| 3 | Otsu, Canny, abertura/fechamento morfológico |
| 4 | `image_dataset_from_directory` (80/20), augmentation, `prefetch` |
| 5 | CNN Sequential (`Conv`×3, Dropout 0.5, sigmoid), treino 12 épocas |
| 6 | Gráficos loss/accuracy e diagnóstico de overfitting |

## Decisões técnicas (resumo)

- **Otsu** no lugar de limiar fixo: a iluminação da linha muda; Otsu adapta pelo histograma.  
- **medianBlur** vs gaussiano: impulsos (sujeira/sensor) vs ruído aditivo suave.  
- **Augmentation no grafo** (`RandomFlip` / `RandomRotation` / `RandomZoom`): simula pose na esteira e viaja no `.keras` exportado.  
- **`prefetch(AUTOTUNE)`**: sobrepõe I/O e treino; sem isso a GPU/CPU fica ociosa esperando JPEG.  
- **Dropout 0.5** antes da saída densa: o `Flatten` explode parâmetros num dataset ~1k imagens.  
- **sigmoid + `binary_crossentropy`**: duas classes mutuamente exclusivas em uma saída.

## Reproduzir o histórico Git (sprints)

O script `setup_git_history.sh` reescreve o histórico local com commits Conventional Commits e datas espaçadas (rotina de 2–3 dias). Use apenas se quiser esse histórico simulado:

```bash
# Git Bash / WSL / Linux / macOS
bash setup_git_history.sh
```

## Como avaliar o resultado

1. Rode o notebook de ponta a ponta.  
2. Confira os painéis OpenCV (defeito vs OK).  
3. Após o treino, leia o diagnóstico automático das curvas na Sprint 6.  
4. Em produção, complemente accuracy com **recall da classe defeituosa** (falso OK é o erro caro).

## Licença e dados

Código do mini-projeto para fins educacionais. O dataset permanece sujeito aos termos da fonte original (Kaggle / autores do *Casting Product Image Data*).

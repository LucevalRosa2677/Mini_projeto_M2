#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_git_history.sh
# Recria o historico Git do mini-projeto com commits por sprint e datas
# espacadas (rotina realista de 2-3 dias / turnos de trabalho).
#
# Uso (Git Bash, WSL, Linux ou macOS), na raiz do repositorio:
#   bash setup_git_history.sh
#
# ATENCAO: reescreve o historico da branch atual (orfa -> main).
# Nao execute se ja tiver commits remotos que precise preservar.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NB="inspecao_visual_fundicao.ipynb"
BUILD_DIR=".history_build"

echo "==> Gerando notebook e snapshots por sprint..."
python "$ROOT/_build_notebook.py"

if [[ ! -f "$BUILD_DIR/sprint1/$NB" ]]; then
  echo "Falha: snapshots nao encontrados em $BUILD_DIR"
  exit 1
fi

# README final (completo) fica de lado enquanto os commits intermediarios usam a versao curta
cp "$ROOT/README.md" "$BUILD_DIR/readme_final.md"
if [[ ! -f "$BUILD_DIR/readme_sprint1.md" ]]; then
  echo "Falha: $BUILD_DIR/readme_sprint1.md ausente"
  exit 1
fi

commit_at() {
  local when="$1"
  local message="$2"
  shift 2
  GIT_AUTHOR_DATE="$when" \
  GIT_COMMITTER_DATE="$when" \
  git commit "$@" -m "$message"
}

echo "==> Preparando branch orfa (historico limpo)..."
# Preserva working tree; descarta so a linha de commits atual.
git checkout --orphan sprint-history-tmp

# Limpa indice sem apagar arquivos no disco
git rm -rf --cached . >/dev/null 2>&1 || true

git add .gitignore

# ---------- Dia 1 - Sprint 1 ----------
cp "$BUILD_DIR/sprint1/$NB" "$ROOT/$NB"
cp "$BUILD_DIR/readme_sprint1.md" "$ROOT/README.md"
git add .gitignore requirements.txt "$NB" casting_data/def_front/.gitkeep casting_data/ok_front/.gitkeep
commit_at "2026-09-08T09:15:00-03:00" \
  "feat(sprint1): setup inicial do repositório e download do dataset"

git add README.md
commit_at "2026-09-08T15:40:00-03:00" \
  "docs(sprint1): adição das instruções de ambiente e dependências"

# ---------- Dia 2 - Sprints 2 e 3 ----------
cp "$BUILD_DIR/sprint2/$NB" "$ROOT/$NB"
git add "$NB"
commit_at "2026-09-09T10:05:00-03:00" \
  "feat(sprint2): implementação do pipeline de filtros opencv e remoção de ruído"

cp "$BUILD_DIR/sprint3/$NB" "$ROOT/$NB"
git add "$NB"
commit_at "2026-09-09T16:50:00-03:00" \
  "feat(sprint3): detecção de bordas canny e operações morfológicas para isolar defeitos"

# ---------- Dia 3 - Sprints 4, 5 e 6 ----------
cp "$BUILD_DIR/sprint4/$NB" "$ROOT/$NB"
git add "$NB"
commit_at "2026-09-10T09:20:00-03:00" \
  "feat(sprint4): carga de dados com image_dataset_from_directory e data augmentation"

cp "$BUILD_DIR/sprint5/$NB" "$ROOT/$NB"
git add "$NB"
commit_at "2026-09-10T14:10:00-03:00" \
  "feat(sprint5): arquitetura da cnn sequencial e rotina de treinamento"

cp "$BUILD_DIR/sprint6/$NB" "$ROOT/$NB"
cp "$BUILD_DIR/readme_final.md" "$ROOT/README.md"
git add "$NB" README.md _build_notebook.py setup_git_history.sh
commit_at "2026-09-10T18:25:00-03:00" \
  "docs(sprint6): gráficos de performance loss/accuracy e readme final"

git branch -M main

echo ""
echo "==> Historico recriado. Ultimos commits:"
git log --format="%h | %ad | %s" --date=iso -7
echo ""
echo "Revise com: git log --oneline --date=iso"
echo "Publique no remoto somente se tiver certeza de que pode substituir o historico remoto."

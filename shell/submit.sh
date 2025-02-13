#!/bin/bash
while getopts "r:n:u:e:" opt; do
  case $opt in
    r) repo=$OPTARG ;;
    n) name=$OPTARG ;;
    u) user=$OPTARG ;;
    e) email=$OPTARG ;;
    *) echo "Invalid option" ;;
  esac
done

echo "Cloning $repo"
git clone $repo
dirname=$(ls -td ./*/ | head -1)
cd $dirname

# add readme and push initial commit
touch README.md

git add README.md
git commit -m "first commit"
git push origin HEAD:main

# make directories
mkdir -p "src/$name" scripts test shell experiments

touch "src/$name/__init__.py"
touch "src/$name/utils.py"

touch scripts/run.py

touch shell/submit.sbatch
touch shell/submit.sh

touch experiments/config.yaml

touch test/test_.py

python -m pip freeze > "$(git rev-parse --show-toplevel)/requirements_test.txt"
# echo "$repo"
# echo "$name"
# echo "$user"
# echo "$email"
curl -s https://raw.githubusercontent.com/dario-coscia/devtools_scicomp_project_2025/refs/heads/main/pyproject.toml -o pyproject.toml
sed -i '' -e "s/INSERT@gmail.com/$email/g" pyproject.toml
sed -i '' -e "s/INSERT/$user/g" pyproject.toml

echo '# data files' >> .gitignore
echo '*.dat' >> .gitignore
echo '*.data' >> .gitignore

git add .
git commit -m "structuring project"
git push origin HEAD:main

for f in *.tar; do
  dname=$(basename "$f" .tar)
  mkdir "$dname"
  tar -xf "$f" -C "$dname"
done
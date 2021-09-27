{ echo "Primo blocco";
  python ./evolving_clustering/script_ubuntu_ubuntu.py --step=10 --first_threshold="0.041" --second_threshold="0.015" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.035" --second_threshold="0.015" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.028" --second_threshold="0.015" --randomly --seed="42" } &
sleep 3
{
  echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.041" --second_threshold="0.020" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.035" --second_threshold="0.020" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.028" --second_threshold="0.020" --randomly --seed="42" ;
} &
sleep 3
{
  echo "Terzo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.041" --second_threshold="0.010" --randomly --seed="42"
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.035" --second_threshold="0.010" --randomly --seed="42"
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.028" --second_threshold="0.010" --randomly --seed="42"
}


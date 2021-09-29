{ echo "Primo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.005" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.007" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.009" --randomly --seed="42"; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.001" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.015" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ; } &
sleep 3
{ echo "Terzo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.017" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.021" --randomly --seed="42";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.003" --randomly --seed="42"; }
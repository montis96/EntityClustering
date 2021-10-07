{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="3" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="80" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="16" --entropy="10";} &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=1 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=30 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=70 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=120 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="10"; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="42" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="42" --entropy="10";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.02" --randomly --seed="42" --entropy="10";} &

basepath=$(cd $(dirname $0) || exit; pwd)
cd $basepath || exit
echo "Working at "$(pwd)"..."

if [ ! -d "./logs" ]; then
     mkdir ./logs
fi


#echo "=============================A_share, stock-stock, RSR, heterograph============================="
#python -u run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
#                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
#                     --result_root "/data/dengjiale/qlib_exp/results/" \
#                     --market "A_share" \
#                     --relation_type "stock-stock" \
#                     --graph_model "RSR" \
#                     --graph_type "heterograph" \
#                     --gpu 1  2>&1 | tee logs/RSR.log
#
#echo "=============================A_share, stock-stock, simpleHGN, heterograph============================="
#python -u run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
#                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
#                     --result_root "/data/dengjiale/qlib_exp/results/" \
#                     --market "A_share" \
#                     --relation_type "stock-stock" \
#                     --graph_model "simpleHGN" \
#                     --graph_type "heterograph" \
#                     --gpu 1  2>&1 | tee logs/simpleHGN.log
#
#echo "=============================A_share, stock-stock, GAT, homograph============================="
#python -u run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
#                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
#                     --result_root "/data/dengjiale/qlib_exp/results/" \
#                     --market "A_share" \
#                     --relation_type "stock-stock" \
#                     --graph_model "GAT" \
#                     --graph_type "homograph" \
#                     --gpu 1  2>&1 | tee logs/GAT.log


for model_type in RSR simpleHGN GAT
do
echo "============================= 对"$model_type"的预测结果进行解释 ============================="
for sparsity in 3 4
do
  echo ">>> 当解释大小限定为"$sparsity"，各解释方法的fidelity分数分别为 <<<"
  for method in effect subgraphx xpath
  do
    score=`tail -n40 logs/$model_type.log | grep $method -a7 | grep "sparsity: "$sparsity -a1 | tail -n1 | awk -F 'score ' '{print $2}' | cut -c1-8`
    case $method in
      subgraphx)
          tab="\t";;
      *)
          tab="\t\t";;
    esac
    echo $method":"$tab$score
  done
done
done
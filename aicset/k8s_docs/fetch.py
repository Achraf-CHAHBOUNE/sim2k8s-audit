import urllib.request,re,html,sys,datetime
pages={
"kube_scheduler":"https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/",
"pod_lifecycle":"https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
"reserve_compute":"https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/",
"multiple_schedulers":"https://kubernetes.io/docs/tasks/extend-kubernetes/configure-multiple-schedulers/",
"manage_resources":"https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/",
"pod_v1_api":"https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/pod-v1/",
"kubelet_config":"https://kubernetes.io/docs/reference/config-api/kubelet-config.v1beta1/",
"resource_metrics":"https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/",
"checkpoint":"https://kubernetes.io/docs/reference/node/kubelet-checkpoint-api/",
"node_status":"https://kubernetes.io/docs/reference/node/node-status/",
"nodes":"https://kubernetes.io/docs/concepts/architecture/nodes/",
}
for k,u in pages.items():
    try:
        t=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=30).read().decode("utf-8","ignore")
    except Exception as e:
        print(k,"FAIL",e);continue
    t=re.sub(r'(?s)<(script|style|nav|header|footer).*?</\1>','',t)
    t=html.unescape(re.sub(r'<[^>]+>',' ',t)); t=re.sub(r'[ \t\r\f\v]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    open(k+".txt","w",encoding="utf-8").write(f"SOURCE: {u}\nRETRIEVED: {datetime.date.today()}\n\n"+t)
    print(k,len(t))

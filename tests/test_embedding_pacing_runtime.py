"""Disposable runtime checks for the native embedding pacing primitive."""
from __future__ import annotations
import importlib.util, os, subprocess, sys, types
from pathlib import Path
import pytest
SOURCE = Path(__file__).parents[1] / "src" / "paperless_ai" / "indexing.py"
EXCEPTIONS = SOURCE.with_name("exceptions.py")

def load_indexing(data_dir: Path):
    from django.conf import settings
    if not settings.configured:
        settings.configure(DATA_DIR=str(data_dir), USE_TZ=True, SECRET_KEY="test")
    docs=types.ModuleType("documents"); models=types.ModuleType("documents.models")
    models.Document=type("Document",(),{}); models.PaperlessTask=type("PaperlessTask",(),{})
    utils=types.ModuleType("documents.utils")
    class GenericStub:
        @classmethod
        def __class_getitem__(cls,item): return cls
    utils.IterWrapper=utils.QuerySetStream=GenericStub; utils.identity=lambda *a,**k: None
    paperless=types.ModuleType("paperless"); config=types.ModuleType("paperless.config"); config.AIConfig=type("AIConfig",(),{})
    db=types.ModuleType("paperless_ai.db"); db.db_connection_released=lambda: None
    embedding=types.ModuleType("paperless_ai.embedding"); embedding.build_llm_index_text=embedding.get_configured_model_name=embedding.get_embedding_model=lambda *a,**k: None
    exception_spec=importlib.util.spec_from_file_location("paperless_ai.exceptions",EXCEPTIONS); exception_module=importlib.util.module_from_spec(exception_spec); exception_spec.loader.exec_module(exception_module)
    sys.modules.update({"documents":docs,"documents.models":models,"documents.utils":utils,"paperless":paperless,"paperless.config":config,"paperless_ai.db":db,"paperless_ai.embedding":embedding,"paperless_ai.exceptions":exception_module})
    spec=importlib.util.spec_from_file_location("paperless_ai.indexing_runtime",SOURCE); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def worker(argv):
    data_dir,model,clock,size=argv; module=load_indexing(Path(data_dir)); module.time.time=lambda: float(clock)
    try: module._reserve_embedding_capacity(model,["x"*int(size)])
    except Exception as exc: print(f"DEFER:{getattr(exc,'retry_after',None)}"); return 2
    print("RESERVED"); return 0

@pytest.mark.skipif(not EXCEPTIONS.exists(),reason="Paperless source checkout unavailable")
def test_cross_process_spacing_and_tpm_window(tmp_path):
    if len(sys.argv)>=5 and sys.argv[1]=="--worker": raise SystemExit(worker(sys.argv[2:6]))
    env=os.environ.copy(); env["PYTHONPATH"]=str(SOURCE.parents[1])
    def invoke(model,clock,size):
        return subprocess.run([sys.executable,str(Path(__file__)),"--worker",str(tmp_path),model,str(clock),str(size)],cwd=SOURCE.parents[1],capture_output=True,text=True,env=env,check=False)
    first=invoke("spacing",1000,4000); second=invoke("spacing",1000,1)
    assert (first.returncode,first.stdout.strip())==(0,"RESERVED")
    assert second.returncode==2 and second.stdout.startswith("DEFER:7.5")
    assert invoke("spacing",1060,1).stdout.strip()=="RESERVED"
    full=invoke("tpm",2000,96000); over=invoke("tpm",2007.5,1); after=invoke("tpm",2060,1)
    assert full.stdout.strip()=="RESERVED"
    assert over.returncode==2 and float(over.stdout.split(":",1)[1])>=52.5
    assert after.stdout.strip()=="RESERVED"

def _reserve(module):
    try: module._reserve_embedding_capacity("simultaneous",["x"*4000])
    except module.EmbeddingRateLimitError: return "DEFER"
    return "RESERVED"

def test_simultaneous_workers_have_one_reservation(tmp_path):
    module=load_indexing(tmp_path); module.time.time=lambda:3000.0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(lambda _: _reserve(module),range(2)))
    assert sorted(results)==["DEFER","RESERVED"]

if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--worker":
    raise SystemExit(worker(sys.argv[2:6]))

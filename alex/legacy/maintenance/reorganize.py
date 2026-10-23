from pathlib import Path
import shutil,json,hashlib
import yaml
root=Path(__file__).resolve().parents[2]
records=[]
def move(src,dst):
 src=root/src;dst=root/dst
 if not src.exists():return
 if dst.exists():raise RuntimeError(f'Already exists: {dst}')
 files=[src] if src.is_file() else [p for p in src.rglob('*') if p.is_file()]
 before={str(p.relative_to(src)) if p!=src else '.':(p.stat().st_ino,p.stat().st_size) for p in files}
 dst.parent.mkdir(parents=True,exist_ok=True)
 src.rename(dst)
 for rel,stat in before.items():
  p=dst if rel=='.' else dst/rel
  assert (p.stat().st_ino,p.stat().st_size)==stat
 records.append({'from':str(src.relative_to(root)),'to':str(dst.relative_to(root)),'files':len(files),'bytes':sum(x[1] for x in before.values())})
base=yaml.safe_load((root/'solver_python/config.yaml').read_text())
mag=yaml.safe_load((root/'results/binaries/magnets_2x3_re260/config.resolved.yaml').read_text())
for name,data in [('pulsation_re260',base),('pulsation_re260_stabilized',base),('magnets_2x3_re260',mag)]:
 data=__import__('copy').deepcopy(data)
 data.pop('experiment',None);data['output'].pop('root',None)
 if data['forcing']['type']=='custom':data['forcing']['file']='../../solver_python/sources/harmonics.py'
 if name!='pulsation_re260_stabilized':data.pop('stabilization',None)
 else:data['stabilization']['file']='../../results/pulsation_re260/linear_stability/stabilization.npz'
 dest=root/'configs'/name;dest.mkdir(parents=True)
 (dest/f'{name}.yaml').write_text('# Имя этого YAML определяет папку results/<имя>/.\n'+yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
 # These are explicitly requested experiment descriptions, not general project docs.
 text={'pulsation_re260':'Базовый расчёт с гармоническим источником, Re=260, 201×101, область 1×0.5. Миллион шагов по 1e-5 до псевдовремени 10. Неподвижные стенки, прилипание, без придонного трения. Служит для поиска стационара и неустойчивых мод.',
 'pulsation_re260_stabilized':'Та же задача, что pulsation_re260, с обратной связью −P(ω−ω*) в правой части уравнения завихренности; gain=1. Файл стационара и базиса создаётся анализом базового расчёта. Это управление в псевдовремени, а не физическое граничное управление.',
 'magnets_2x3_re260':'Модельная магнитная установка: 2 столбца × 3 ряда, Re=260. Координаты, d и B заданы в СИ в YAML. Это собственная дипольная модель, не реконструкция параметров статьи. B — поле одного диполя на его оси в плоскости модели; d — расстояние от точки диполя до дна. Плотность тока задаётся в А/м².'}[name]
 (dest/'README.md').write_text(f'# {name}\n\n{text}\n\nРезультаты: `alex/results/{name}/`. Схема всегда факторизованная, якобиан Аракавы. Параметры запуска и потоки находятся в YAML.\n')
# Active Python utilities become self-contained; old copies are archived below.
shutil.copy2(root/'common/snapshot_io.py',root/'solver_python/snapshot_io.py')
for old,new in [('plot_fields.py','plot_fields.py'),('make_videos.py','make_videos.py'),('find_stationary.py','find_stationary.py')]:
 shutil.copy2(root/'scripts/python'/old,root/'solver_python'/new)
active={'pulsation_re260','pulsation_re260_stabilized','magnets_2x3_re260'}
for category in ['binaries','figures','videos','tables','stationary','newton','linear_stability']:
 parent=root/'results'/category
 for entry in list(parent.iterdir()):
  if entry.name in active:move(Path('results')/category/entry.name,Path('results')/entry.name/category)
 # All remaining older results and metadata are archived, including empty placeholders.
 move(Path('results')/category,Path('legacy/results_by_category')/category)
for name in ['cpu','gpu','common','scripts','pipeline','build','data','docs','linear_stability','plots','paper','presentation','RUN','Makefile','requirements.txt','codex_handoff.zip','.DS_Store']:
 move(Path(name),Path('legacy')/name)
move(Path('solver_python/config.yaml'),Path('legacy/configs/solver_python_original.yaml'))
move(Path('results/.DS_Store'),Path('legacy/results_by_category/.DS_Store'))
(root/'legacy/maintenance/migration.json').write_text(json.dumps(records,indent=2))
print('Moved',sum(r['files'] for r in records),'files, verified inode and size; no scientific output rewritten.')

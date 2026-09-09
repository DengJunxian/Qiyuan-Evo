"""Capture real Streamlit pages via playwright-cli and bind images to experiments.

Requires Node.js/npm and a running app. No market results or screenshots are synthesized.
"""
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[1]


def capture(base_url, destination):
    cases=json.loads((ROOT/'benchmarks/competition/reference/before_after.json').read_text())
    by_task={c['baseline']['task']['task_type']:c for c in cases}
    policy='policy_report';history='historical_analysis'
    shots=[
        ('01-overview.png','自演进总览','自演进驾驶舱',None,policy,'一次任务，让团队学会更好的分工。',
         '同一任务的验证完成度、规则评分和执行成员变化。'),
        ('02-collaboration.png','初轮分工与审查','协作过程',None,policy,'团队如何协作，审查又发现了什么',
         '规划员拆分任务，成员沿依赖移交结果；审查发现对照与独立复核遗漏。'),
        ('03-comparison.png','优化前后与组织变化','演进实验','compare',policy,'同一任务，团队改进了什么',
         '相同任务、种子与数据，比较加入独立复核员前后的验证结果。'),
        ('04-prompt.png','模型优化提示词与工具策略','演进实验','prompt',policy,'同一任务，团队改进了什么',
         'DeepSeek 根据审查反馈生成提示词；执行契约被下一轮规划实际使用。'),
        ('05-memory.png','历史经验进入下一轮规划','演进实验','memory',policy,'同一任务，团队改进了什么',
         '查看规划员本轮引用的经验与采用的团队配置。'),
        ('06-benchmarks.png','三类可复现实验','三类任务',None,policy,'实验案例',
         '报告生成、数据分析、任务规划共用同一套团队执行与演进流程。'),
        ('07-parallel-team.png','历史任务的并行分工','演进实验','compare',history,'同一任务，团队改进了什么',
         '历史分析员与量化分析员加入，审查等待两个并行任务完成。'),
        ('08-runtime.png','openJiuwen 调用与复现','技术与复现',None,policy,'运行环境与实验资料',
         '已安装版本、本次工作流完成状态、模型调用和证据包下载入口。'),
    ]
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    work=ROOT/'output/playwright';work.mkdir(parents=True,exist_ok=True)
    cli=['npx','--yes','--package','@playwright/cli','playwright-cli','-s=qiyuan-gallery']
    def run(args):
        result=subprocess.run(cli+args,cwd=ROOT,text=True,capture_output=True,timeout=70)
        if result.returncode or '### Error' in result.stdout:
            raise RuntimeError((result.stdout+result.stderr)[-2500:])
        return result.stdout
    run(['open',base_url]);run(['resize','1440','1100'])
    records=[]
    try:
        for filename,title,page,view,family,heading,description in shots:
            run(['resize','1440','1100'])
            case=by_task[family]
            params={'page':page,'artifact_run':case['baseline']['run_id']}
            if view:params['view']=view
            url=base_url.rstrip('/')+'/?'+urlencode(params)
            run(['goto',url])
            # Fresh snapshot is also retained for page/locator diagnosis.
            (work/(filename+'.snapshot.txt')).write_text(run(['snapshot']))
            target=destination/filename
            code='''async (page) => {
                const errors=[];
                page.on('pageerror',e=>errors.push(e.message));
                await page.getByRole('heading', {name: HEADING,exact:true}).waitFor({timeout:45000});
                await page.locator('[data-testid="stMainBlockContainer"]').waitFor();
                await page.evaluate(()=>document.fonts.ready);
                await page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').waitFor({timeout:45000});
                if (await page.locator('[data-testid="stException"]').count()) throw new Error('Streamlit exception');
                const panel=page.locator('[data-testid="stMainBlockContainer"]');
                const box=await panel.boundingBox();
                if (box.height<300) throw new Error('Page has not finished rendering');
                // Streamlit virtualizes off-screen blocks. Fit the full panel
                // into the viewport so the screenshot contains painted content.
                await page.setViewportSize({width:1440,height:Math.ceil(box.height)+30});
                for (const frame of page.frames().slice(1)) {
                    if (await frame.locator('svg').count()) await frame.locator('svg').waitFor();
                }
                await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
                await panel.screenshot({path:TARGET,animations:'disabled'});
                if(errors.length) throw new Error(errors.join('; '));
                return {url:page.url(),height:box.height,width:box.width};
            }'''.replace('HEADING',json.dumps(heading,ensure_ascii=False)).replace('TARGET',json.dumps(str(target)))
            script=work/'capture_page.js';script.write_text(code)
            run(['run-code','--filename='+str(script)])
            optimizer=case['evolution_history'][0]['optimizer']
            records.append({'file':filename,'title':title,'description':description,'page':page,'view':view,
                'experiment_id':case['experiment_id'],'baseline_run_id':case['baseline']['run_id'],
                'backend':case['evolved']['runtime']['backend'],'model':optimizer.get('model','none'),
                'optimizer_backend':optimizer['backend'],'source_hash':case['evolved']['runtime']['source_hash'],
                'captured_at':datetime.now(timezone.utc).isoformat(),'sha256':sha256(target.read_bytes()).hexdigest()})
            print('Captured '+filename,flush=True)
    finally:
        run(['close'])
    (destination/'manifest.json').write_text(json.dumps({'schema_version':1,'type':'completed_experiment_screenshots',
        'screenshots':records},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://localhost:8502')
    parser.add_argument('--output',type=Path,default=ROOT/'static/competition_gallery')
    args=parser.parse_args();capture(args.url,args.output)

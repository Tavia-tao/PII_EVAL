// 纯内存DOM单元测试：不启动浏览器/HTTP服务，不加载网络或本地资源。
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {JSDOM} from 'jsdom';

function fixture() {
  const rows=Array.from({length:85},(_,i)=>({sample_id:`st_${String(i).padStart(3,'0')}`,
    track:i%2?'complex_file':'standard_text',layout:i%2?'scan':'pure_text',difficulty:'easy',
    tp:1,fp:i===4?1:0,fn:0,f1:i===4?.667:1,partial:0,latency_ms:5,route:'student',
    route_reason:'default',execution_error:false,trace_id:'a'.repeat(32),
    by_type:{[i%2?'PHONE':'NAME']:{recall:1}},stages:{S1_parse:{ok:true,elapsed_ms:1,metrics:{blocks:1}}},
    gold_spans:[{start:3,end:6,type:'NAME'}],pred_spans:[{start:4,end:6,type:'NAME'}],fp_spans:[],fn_spans:[]}));
  const m={n:42,f1:1,by_type:{NAME:{recall:1}},high_risk_recall:1,false_delete:0,risk_score:0,latency_p95:.1,review_rate:0};
  const data={report:{meta:{run_tag:'test',synthetic_only:true,model_alias:'demo-student'},
    tracks:{standard_text:m,complex_file:m},operational:m,ci:{standard_text:{lo:1,hi:1},complex_file:{lo:1,hi:1}},
    rows,by_stage:{S1_parse:{mean_elapsed_ms:1}}},
    samples:Object.fromEntries(rows.map(r=>[r.sample_id,{text:'姓名：😀张三，</script><img src=x onerror="globalThis.hacked=1">'}]))};
  const html=readFileSync(new URL('../src/report_template.html',import.meta.url),'utf8')
    .replace('__REPORT_DATA__',JSON.stringify(data).replaceAll('<','\\u003c'));
  let csvBlob;
  const dom=new JSDOM(html,{runScripts:'dangerously',beforeParse(w){
    w.HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','')};
    w.HTMLDialogElement.prototype.close=function(){this.removeAttribute('open')};
    w.HTMLAnchorElement.prototype.click=function(){};
    w.Blob=globalThis.Blob;w.URL.createObjectURL=b=>{csvBlob=b;return 'blob:test'};w.URL.revokeObjectURL=()=>{};
  }});
  return {dom,doc:dom.window.document,csv:()=>csvBlob};
}

test('pagination and filtering operate on actual rows',()=>{
  const {dom,doc}=fixture();
  assert.equal(doc.querySelectorAll('#rows tr').length,40);
  doc.querySelector('#next').click();assert.match(doc.querySelector('#range').textContent,/41–80/);
  const query=doc.querySelector('#query');query.value='st_004';query.dispatchEvent(new dom.window.Event('input'));
  assert.equal(doc.querySelectorAll('#rows tr').length,1);
  assert.equal(doc.querySelector('#rows button').textContent,'st_004');
  query.value='';query.dispatchEvent(new dom.window.Event('input'));
  doc.querySelector('#failed').click();assert.equal(doc.querySelectorAll('#rows tr').length,1);
  dom.window.close();
});

test('detail renders Unicode spans safely and closes',()=>{
  const {dom,doc}=fixture();doc.querySelector('#rows button').click();
  assert.equal(doc.querySelector('#detail').hasAttribute('open'),true);
  assert.equal(doc.querySelector('#gold mark').textContent,'😀张三');
  assert.equal(doc.querySelector('#pred mark').textContent,'张三');
  assert.equal(doc.querySelectorAll('#gold img').length,0);
  assert.equal(dom.window.hacked,undefined);
  doc.querySelector('#close').click();assert.equal(doc.querySelector('#detail').hasAttribute('open'),false);
  dom.window.close();
});

test('CSV export contains the selected data rather than only visible page',async()=>{
  const {dom,doc,csv}=fixture();doc.querySelector('#csv').click();
  const text=await csv().text();assert.match(text,/st_084/);assert.equal(text.split('\r\n').length,86);
  dom.window.close();
});

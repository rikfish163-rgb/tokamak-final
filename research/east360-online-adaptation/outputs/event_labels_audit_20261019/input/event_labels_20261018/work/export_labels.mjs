import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook} from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const payload=JSON.parse(await fs.readFile(path.join(here,'labels_payload.json'),'utf8'));
const wb=Workbook.create();
const sheet=wb.worksheets.add('Event labels');
const matrix=[payload.header,...payload.rows];
const range=sheet.getRangeByIndexes(0,0,matrix.length,payload.header.length);
range.values=matrix;
wb.recalculate();
const check=await wb.inspect({kind:'region',sheetId:sheet.name,range:'A1:H6',maxChars:1800,tableMaxRows:6,tableMaxCols:8});
await fs.writeFile(path.join(here,'artifact_inspection.ndjson'),check.ndjson);
const result=range.values;
if(JSON.stringify(result)!==JSON.stringify(matrix)) throw new Error('Typed cell round-trip changed values');
// CSV is serialized from the authored worksheet values; no formulas or formatting are needed.
const field=(v)=>v===null || v===undefined ? '' : /[",\r\n]/.test(String(v)) ? '"'+String(v).replaceAll('"','""')+'"' : String(v);
const csv='\uFEFF'+result.map(row=>row.map(field).join(',')).join('\r\n')+'\r\n';
await fs.writeFile(path.join(here,'..','JTEXT_event_labels.csv'),csv,'utf8');
console.log(JSON.stringify({rows:result.length-1,columns:result[0].length,output:'JTEXT_event_labels.csv'}));

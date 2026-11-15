"""Opt-in browser notifications mounted in the app document (no iframe)."""

import hashlib

import streamlit as st

from app import api_client

_NOTIFICATIONS = st.components.v2.component(
    "pravah_nepal_notifications",
    html='<div><button id="enable">Enable browser notifications</button> <button id="disable">Turn off</button><p id="status" role="status"></p></div>',
    css='button{border:1px solid #263448;border-radius:8px;background:#172335;color:#F3F6FA;padding:9px 14px;cursor:pointer;margin-right:8px}p{color:#A4B2C5;font:12px system-ui}',
    js=r"""
export default function(component) {
  const {data, parentElement} = component;
  const status = parentElement.querySelector('#status');
  const enable = parentElement.querySelector('#enable');
  const disable = parentElement.querySelector('#disable');
  const prefix = 'pravah-nepal-notifications-' + data.namespace;
  let enabled = false, seen = new Set();
  try {
    enabled = localStorage.getItem(prefix + '-enabled') === 'yes';
    seen = new Set(JSON.parse(localStorage.getItem(prefix + '-seen') || '[]'));
  } catch (_) {}
  const supported = typeof Notification !== 'undefined' && window.isSecureContext;
  const persist = () => {try {
    localStorage.setItem(prefix + '-enabled', enabled ? 'yes' : 'no');
    localStorage.setItem(prefix + '-seen', JSON.stringify([...seen].slice(-500)));
  } catch (_) {}};
  const render = () => {
    enable.disabled = !supported || Notification.permission === 'denied';
    disable.disabled = !enabled;
    status.textContent = !supported ? 'Browser notifications require a supported browser and HTTPS or localhost.' :
      Notification.permission === 'denied' ? 'Permission denied. Change this site’s browser settings to enable notifications.' :
      enabled && Notification.permission === 'granted' ? 'Browser notifications enabled while this dashboard is open.' :
      'Dashboard alerts are active. Browser notifications are optional.';
  };
  enable.onclick = async () => {
    if (!supported) return;
    const permission = await Notification.requestPermission();
    enabled = permission === 'granted';
    data.alerts.forEach(a => seen.add(a.id)); // Existing history is not a new alert.
    persist(); render();
  };
  disable.onclick = () => {enabled = false; persist(); render();};
  const initialized = (() => {try {return localStorage.getItem(prefix + '-initialized') === 'yes';}catch (_) {return false;}})();
  for (const alert of data.alerts) {
    if (initialized && enabled && supported && Notification.permission === 'granted' && !seen.has(alert.id)) {
      try {new Notification('PRAVAH · Nepal source watch', {body: alert.name + ': ' + alert.message, tag: alert.id});} catch (_) {}
    }
    seen.add(alert.id);
  }
  try {localStorage.setItem(prefix + '-initialized', 'yes');}catch (_) {}
  persist(); render();
}
""",
)


def render_notifications(alerts):
    namespace = hashlib.sha256(api_client.api_url().encode()).hexdigest()[:12]
    _NOTIFICATIONS(data={"alerts": alerts, "namespace": namespace}, key="pv_nepal_notifications")
